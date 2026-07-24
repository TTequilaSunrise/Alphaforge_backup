"""StockData backed by BigQuant DAI (cn_stock_bar1d).

Public API matches the old qlib StockData so AlphaForge training code
(train_AFF / combine_AFF / expression.evaluate) does not need changes.

Requires conda env with ``bigquant`` (preferred) or ``bigquantdai``, and
read access to ``cn_stock_bar1d`` + ``cn_stock_index_component``.
"""

from enum import IntEnum
from typing import List, Union, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch

from alphagen_qlib.bigquant_backend import (
    fetch_bar1d,
    fetch_trading_calendar,
    resolve_instruments,
    to_qlib_code,
)

# Optional BigAlpha 2026 competition backend (1min -> daily aggregation).
_BIGALPHA_BACKEND = "bigalpha2026"


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


def change_to_raw_min(features: list[str]) -> list[str]:
    result = []
    for feature in features:
        if feature in ["$vwap"]:
            result.append("$money/$volume")
        elif feature in ["$volume"]:
            result.append(f"{feature}/100000")
        else:
            result.append(feature)
    return result


def change_to_raw(features: list[str]) -> list[str]:
    result = []
    for feature in features:
        if feature in ["$open", "$close", "$high", "$low", "$vwap"]:
            result.append(f"{feature}*$factor")
        elif feature in ["$volume"]:
            result.append(f"{feature}/$factor/1000000")
        else:
            raise ValueError(f"feature {feature} not supported")
    return result


class StockData:
    _backend_initialized: bool = False

    def __init__(
        self,
        instrument: Union[str, List[str]],
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cuda:0"),
        raw: bool = False,
        qlib_path: Union[str, Dict] = "",
        freq: str = "day",
        backend: str = "bigalpha2026",
        datasources: Optional[Dict[str, str]] = None,
    ) -> None:
        # qlib_path kept for call-site compatibility; unused.
        _ = qlib_path
        self._init_backend()
        self.df_bak = None
        self.raw = raw
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq
        # backend: "bigalpha2026" (competition 1min->daily) | "cn_bar1d" (cn_stock_bar1d)
        self.backend = backend
        self.datasources = datasources
        if self.freq != "day":
            raise NotImplementedError(
                "StockData tensor API is day-frequency. "
                "Use alphagen_qlib.bigalpha2026.fetch_bar1m for raw 1-minute+LOB."
            )
        self.data, self._dates, self._stock_ids = self._get_data()

    @classmethod
    def _init_backend(cls) -> None:
        if cls._backend_initialized:
            return
        from alphagen_qlib.bigquant_backend import ensure_login

        ensure_login()
        cls._backend_initialized = True

    def _expand_query_window(self) -> tuple[str, str, pd.DatetimeIndex]:
        pad_start = (
            pd.Timestamp(self._start_time)
            - pd.Timedelta(days=int(self.max_backtrack_days * 2) + 40)
        ).strftime("%Y-%m-%d")
        pad_end = (
            pd.Timestamp(self._end_time)
            + pd.Timedelta(days=int(self.max_future_days * 2) + 40)
        ).strftime("%Y-%m-%d")
        calendar = fetch_trading_calendar(pad_start, pad_end)
        start_ts = pd.Timestamp(self._start_time)
        end_ts = pd.Timestamp(self._end_time)
        start_index = int(calendar.searchsorted(start_ts))
        end_index = int(calendar.searchsorted(end_ts))
        if end_index >= len(calendar) or calendar[end_index] != end_ts:
            end_index -= 1
        if start_index < self.max_backtrack_days:
            raise ValueError(
                f"Not enough trading days before {self._start_time} "
                f"for max_backtrack_days={self.max_backtrack_days}"
            )
        if end_index + self.max_future_days >= len(calendar):
            raise ValueError(
                f"Not enough trading days after {self._end_time} "
                f"for max_future_days={self.max_future_days}"
            )
        real_start = calendar[start_index - self.max_backtrack_days]
        real_end = calendar[end_index + self.max_future_days]
        used_calendar = calendar[
            start_index - self.max_backtrack_days : end_index + self.max_future_days + 1
        ]
        return (
            real_start.strftime("%Y-%m-%d"),
            real_end.strftime("%Y-%m-%d"),
            used_calendar,
        )

    def _feature_series(self, bars: pd.DataFrame, feat: FeatureType) -> pd.Series:
        has_factor = "adjust_factor" in bars.columns
        if has_factor:
            factor = bars["adjust_factor"].replace(0, np.nan)
        else:
            factor = pd.Series(1.0, index=bars.index)
        vol = bars["volume"].replace(0, np.nan)
        vwap_raw = bars["amount"] / vol

        # cn_stock_bar1d: OHLC already 后复权; bigalpha daily agg: use as-is.
        # AlphaForge training uses raw=True (old qlib: price*$factor, volume/$factor/1e6).
        if self.raw:
            mapping = {
                FeatureType.OPEN: bars["open"],
                FeatureType.CLOSE: bars["close"],
                FeatureType.HIGH: bars["high"],
                FeatureType.LOW: bars["low"],
                FeatureType.VOLUME: bars["volume"] / factor / 1_000_000.0,
                FeatureType.VWAP: vwap_raw * factor,
            }
        else:
            mapping = {
                FeatureType.OPEN: bars["open"] / factor,
                FeatureType.CLOSE: bars["close"] / factor,
                FeatureType.HIGH: bars["high"] / factor,
                FeatureType.LOW: bars["low"] / factor,
                FeatureType.VOLUME: bars["volume"] / 1_000_000.0,
                FeatureType.VWAP: vwap_raw,
            }
        return mapping[feat]

    def _load_bars_and_calendar(
        self,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
        """Return (bars_df, used_calendar, instrument_codes_bq)."""
        if self.backend == _BIGALPHA_BACKEND:
            from alphagen_qlib.bigalpha2026 import (
                fetch_bar1m_as_daily,
                list_instruments_union,
            )

            # Pad calendar with natural days then trim after load.
            pad_start = (
                pd.Timestamp(self._start_time)
                - pd.Timedelta(days=int(self.max_backtrack_days * 2) + 40)
            ).strftime("%Y-%m-%d 00:00:00")
            pad_end = (
                pd.Timestamp(self._end_time)
                + pd.Timedelta(days=int(self.max_future_days * 2) + 40)
            ).strftime("%Y-%m-%d 23:59:59")

            name = (
                self._instrument.strip().lower()
                if isinstance(self._instrument, str)
                else ""
            )
            if name in {"csi1000", "csi300", "csi500", "csi800"} or name == "":
                # Competition instruments table is CSI1000 PIT membership.
                instruments = list_instruments_union(
                    pad_start, pad_end, self.datasources
                )
            elif isinstance(self._instrument, list):
                instruments = list(self._instrument)
            else:
                instruments = [self._instrument]

            bars = fetch_bar1m_as_daily(
                pad_start, pad_end, instruments=instruments, datasources=self.datasources
            )
            calendar = pd.DatetimeIndex(
                sorted(pd.to_datetime(bars["date"]).unique())
            )
            start_ts = pd.Timestamp(self._start_time).normalize()
            end_ts = pd.Timestamp(self._end_time).normalize()
            start_index = int(calendar.searchsorted(start_ts))
            end_index = int(calendar.searchsorted(end_ts))
            if end_index >= len(calendar) or calendar[end_index] != end_ts:
                end_index -= 1
            if start_index < self.max_backtrack_days:
                raise ValueError(
                    f"Not enough trading days before {self._start_time} "
                    f"for max_backtrack_days={self.max_backtrack_days}"
                )
            if end_index + self.max_future_days >= len(calendar):
                raise ValueError(
                    f"Not enough trading days after {self._end_time} "
                    f"for max_future_days={self.max_future_days}"
                )
            used_calendar = calendar[
                start_index
                - self.max_backtrack_days : end_index
                + self.max_future_days
                + 1
            ]
            return bars, used_calendar, instruments

        # Default: official cn_stock_bar1d path
        real_start, real_end, used_calendar = self._expand_query_window()
        instruments = resolve_instruments(self._instrument, real_start, real_end)
        bars = fetch_bar1d(instruments, real_start, real_end)
        return bars, used_calendar, instruments

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        bars, used_calendar, _instruments = self._load_bars_and_calendar()

        stock_ids_bq = sorted(bars["instrument"].astype(str).unique().tolist())
        matrices: list[np.ndarray] = []
        bak_frames: list[pd.DataFrame] = []

        for feat in self._features:
            series = self._feature_series(bars, feat)
            pivot = (
                pd.DataFrame(
                    {
                        "date": bars["date"],
                        "instrument": bars["instrument"],
                        "value": series,
                    }
                )
                .drop_duplicates(subset=["date", "instrument"], keep="last")
                .pivot(index="date", columns="instrument", values="value")
                .reindex(index=used_calendar, columns=stock_ids_bq)
            )
            bak_frames.append(pivot)
            matrices.append(pivot.to_numpy(dtype=np.float32))

        values = np.stack(matrices, axis=0).transpose(1, 0, 2)
        self.df_bak = pd.concat(
            {
                feat.name.lower(): frame
                for feat, frame in zip(self._features, bak_frames)
            },
            axis=1,
        )

        stock_ids = pd.Index([to_qlib_code(x) for x in stock_ids_bq])
        return (
            torch.tensor(values, dtype=torch.float, device=self.device),
            pd.Index(used_calendar),
            stock_ids,
        )

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def add_data(self, data: torch.Tensor, dates: pd.Index) -> None:
        data = data.to(self.device)
        self.data = torch.cat([self.data, data], dim=0)
        self._dates = pd.Index(self._dates.append(dates))

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(
                f"number of days in the provided tensor ({n_days}) doesn't "
                f"match that of the current StockData ({self.n_days})"
            )
        if self.n_stocks != n_stocks:
            raise ValueError(
                f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                f"match that of the current StockData ({self.n_stocks})"
            )
        if len(columns) != n_columns:
            raise ValueError(
                f"size of columns ({len(columns)}) doesn't match with "
                f"tensor feature count ({data.shape[2]})"
            )
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days :]
        else:
            date_index = self._dates[
                self.max_backtrack_days : -self.max_future_days
            ]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(
            data.detach().cpu().numpy(), index=index, columns=columns
        )
