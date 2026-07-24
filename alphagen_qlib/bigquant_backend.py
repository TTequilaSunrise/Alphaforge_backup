"""BigQuant DAI backend for AlphaForge StockData.

Designed for conda env ``bigquant`` (``import bigquant`` + ``init_from_config``).

Doc reference: https://bigquant.com/wiki/doc/PLSbc1SbZX

Tables used:
- ``cn_stock_bar1d``: A-share daily bars (后复权 OHLC)
- ``cn_stock_index_component``: index constituents
  - column ``instrument`` = index code (e.g. 000300.SH)
  - column ``member_code`` = member stock code (e.g. 000001.SZ)
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Universe name -> index code in cn_stock_index_component.instrument
INDEX_CODE_MAP: dict[str, str] = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
    "csi800": "000906.SH",
    "csi1000": "000852.SH",
}

# Liquid stock used only to build a trading-day calendar from cn_stock_bar1d.
_CALENDAR_INSTRUMENT = "000001.SZ"

_dai: Any | None = None
_logged_in: bool = False


def to_bigquant_code(code: str) -> str:
    """qlib ``sh600000`` / ``sz000001`` -> BigQuant ``600000.SH`` / ``000001.SZ``."""
    code = str(code).strip()
    if "." in code:
        num, mkt = code.split(".", 1)
        return f"{num}.{mkt.upper()}"
    lower = code.lower()
    if lower.startswith(("sh", "sz", "bj")) and len(code) >= 8:
        return f"{code[2:]}.{code[:2].upper()}"
    return code


def to_qlib_code(code: str) -> str:
    """BigQuant ``600000.SH`` -> qlib ``sh600000`` (keep AlphaForge stock_id style)."""
    code = str(code).strip()
    if "." in code:
        num, mkt = code.split(".", 1)
        return f"{mkt.lower()}{num}"
    return code.lower()


def _load_aksk() -> tuple[str, str]:
    ak = os.environ.get("BIGQUANT_AK", "").strip()
    sk = os.environ.get("BIGQUANT_SK", "").strip()
    if ak and sk:
        return ak, sk

    config_path = Path(os.path.expanduser("~/.bigquant/config.json"))
    if not config_path.is_file():
        raise RuntimeError(
            "BigQuant credentials not found. Run `bq auth configure` or set "
            "BIGQUANT_AK / BIGQUANT_SK, or write ~/.bigquant/config.json."
        )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    auth = cfg.get("auth", {})
    ak = str(auth.get("ak", "")).strip()
    sk = str(auth.get("sk", "")).strip()
    if not ak or not sk:
        raise RuntimeError("auth.ak / auth.sk missing in ~/.bigquant/config.json")
    return ak, sk


def ensure_login() -> Any:
    """Init BigQuant SDK once; return module/object that exposes ``query`` / ``.df()``.

    Preferred path (env ``bigquant``)::

        import bigquant
        bigquant.init_from_config()
        dai = bigquant.dai

    Fallback: ``bigquantdai`` with explicit login (e.g. alphaforge env).
    """
    global _dai, _logged_in
    if _logged_in and _dai is not None:
        return _dai

    try:
        import bigquant

        try:
            bigquant.init_from_config()
        except Exception:
            ak, sk = _load_aksk()
            bigquant.init(ak=ak, sk=sk)
        _dai = bigquant.dai
        _logged_in = True
        return _dai
    except ImportError:
        pass

    try:
        from bigquantdai import dai

        ak, sk = _load_aksk()
        host = os.environ.get("BIGQUANT_HOST", "bigquant.com")
        port = int(os.environ.get("BIGQUANT_PORT", "17010"))
        dai.login(ak, sk, host=host, port=port)
        _dai = dai
        _logged_in = True
        return _dai
    except ImportError as exc:
        raise ImportError(
            "Need package `bigquant` (conda env bigquant) or `bigquantdai`. "
            "Install: pip install 'bigquant[all]' -i https://pypi.bigquant.com/simple/"
        ) from exc


def query_df(
    sql: str,
    filters: dict[str, list[Any]] | None = None,
    params: dict[str, Any] | None = None,
    full_db_scan: bool = False,
) -> pd.DataFrame:
    """Wrapper around ``dai.query(...).df()`` with partition filters.

    See DAI doc: filters e.g. ``{"date": ["2023-01-01", "2024-01-01"],
    "instrument": ["000001.SZ"]}``. Partition tables cannot full-scan unless
    ``full_db_scan=True``.
    """
    dai = ensure_login()
    result = dai.query(
        sql,
        full_db_scan=full_db_scan,
        filters=filters or {},
        params=params,
    )
    if hasattr(result, "df"):
        return result.df()
    return result


def resolve_instruments(
    instrument: str | list[str],
    start_time: str,
    end_time: str,
) -> list[str]:
    """Resolve ``csi300`` / code list to BigQuant instrument codes.

    Official constituent pattern::

        SELECT date, member_code AS instrument
        FROM cn_stock_index_component
        WHERE instrument = '000300.SH' AND date = '...'
    """
    if isinstance(instrument, list):
        return [to_bigquant_code(x) for x in instrument]

    name = instrument.strip().lower()
    if name in INDEX_CODE_MAP:
        index_code = INDEX_CODE_MAP[name]
        sql = f"""
        SELECT DISTINCT member_code AS instrument
        FROM cn_stock_index_component
        WHERE instrument = '{index_code}'
        """
        df = query_df(sql, filters={"date": [start_time, end_time]})
        if df is None or df.empty:
            raise RuntimeError(
                f"Empty constituents for {name} ({index_code}) "
                f"between {start_time} and {end_time}. "
                "Check cn_stock_index_component permission / date range."
            )
        return sorted(df["instrument"].astype(str).unique().tolist())

    return [to_bigquant_code(instrument)]


def fetch_trading_calendar(start_time: str, end_time: str) -> pd.DatetimeIndex:
    """Trading days from ``cn_stock_bar1d`` via a liquid stock.

    Index codes are not always present in ``cn_stock_bar1d``; use stock bar dates.
    """
    sql = f"""
    SELECT date
    FROM cn_stock_bar1d
    WHERE instrument = '{_CALENDAR_INSTRUMENT}'
    ORDER BY date
    """
    df = query_df(
        sql,
        filters={
            "date": [start_time, end_time],
            "instrument": [_CALENDAR_INSTRUMENT],
        },
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"Empty trading calendar from cn_stock_bar1d "
            f"({_CALENDAR_INSTRUMENT}, {start_time}~{end_time}). "
            "Check cn_stock_bar1d permission."
        )
    dates = pd.to_datetime(df["date"]).drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def fetch_bar1d(
    instruments: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """Fetch 后复权 daily bars from ``cn_stock_bar1d``.

    Columns: date, instrument, open, high, low, close, volume, amount, adjust_factor.
    OHLC are backward-adjusted per DAI docs.
    """
    if not instruments:
        raise ValueError("instruments is empty")

    chunks: list[pd.DataFrame] = []
    chunk_size = 200
    for i in range(0, len(instruments), chunk_size):
        part = instruments[i : i + chunk_size]
        in_list = ", ".join(f"'{x}'" for x in part)
        # filters carry partition pruning; WHERE keeps SQL explicit for debug.
        sql = f"""
        SELECT
            date,
            instrument,
            open,
            high,
            low,
            close,
            volume,
            amount,
            adjust_factor
        FROM cn_stock_bar1d
        WHERE instrument IN ({in_list})
        ORDER BY date, instrument
        """
        df = query_df(
            sql,
            filters={"date": [start_time, end_time], "instrument": part},
        )
        if df is not None and not df.empty:
            chunks.append(df)

    if not chunks:
        raise RuntimeError(
            f"Empty cn_stock_bar1d for {len(instruments)} instruments "
            f"between {start_time} and {end_time}. "
            "Check table permission / filters."
        )

    out = pd.concat(chunks, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["instrument"] = out["instrument"].astype(str)
    for col in ["open", "high", "low", "close", "volume", "amount", "adjust_factor"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
