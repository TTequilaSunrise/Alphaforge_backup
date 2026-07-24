"""BigAlpha 2026 competition data API (CSI1000 / 1min+LOB / PIT financial).

Target env: conda ``bigquant`` + ``bigquant.init_from_config()``.

Default physical tables (platform may remap via ``datasources`` dict)::

    bar1m      -> bigalpha_2026_stock_bar1m
    financial  -> bigalpha_2026_financial
    instruments-> bigalpha_2026_instruments

Training-set contract (user spec):
- Universe: CSI1000 constituents at each historical date
- Range: 2019-01-01 .. 2024-12-31
- Bars: 1-minute K + 10-level order book
- Financials: PIT (``category='lf' AND shift=0``)
"""

from typing import Any

import pandas as pd

from alphagen_qlib.bigquant_backend import ensure_login, query_df

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASOURCES: dict[str, str] = {
    "bar1m": "bigalpha_2026_stock_bar1m",
    "financial": "bigalpha_2026_financial",
    "instruments": "bigalpha_2026_instruments",
}

TRAIN_START = "2019-01-01 00:00:00"
TRAIN_END = "2024-12-31 23:59:59"

# Minute + LOB columns (user schema). ``close`` is not listed; use ``price`` as last.
BAR1M_COLUMNS: list[str] = [
    "date",
    "instrument",
    "time",
    "trading_day",
    "pre_close",
    "open",
    "high",
    "low",
    "price",
    "ask_price1",
    "ask_price2",
    "ask_price3",
    "ask_price4",
    "ask_price5",
    "ask_price6",
    "ask_price7",
    "ask_price8",
    "ask_price9",
    "ask_price10",
    "ask_volume1",
    "ask_volume2",
    "ask_volume3",
    "ask_volume4",
    "ask_volume5",
    "ask_volume6",
    "ask_volume7",
    "ask_volume8",
    "ask_volume9",
    "ask_volume10",
    "bid_price1",
    "bid_price2",
    "bid_price3",
    "bid_price4",
    "bid_price5",
    "bid_price6",
    "bid_price7",
    "bid_price8",
    "bid_price9",
    "bid_price10",
    "bid_volume1",
    "bid_volume2",
    "bid_volume3",
    "bid_volume4",
    "bid_volume5",
    "bid_volume6",
    "bid_volume7",
    "bid_volume8",
    "bid_volume9",
    "bid_volume10",
    "bid_num_orders1",
    "bid_num_orders2",
    "bid_num_orders3",
    "bid_num_orders4",
    "bid_num_orders5",
    "bid_num_orders6",
    "bid_num_orders7",
    "bid_num_orders8",
    "bid_num_orders9",
    "bid_num_orders10",
    "ask_num_orders1",
    "ask_num_orders2",
    "ask_num_orders3",
    "ask_num_orders4",
    "ask_num_orders5",
    "ask_num_orders6",
    "ask_num_orders7",
    "ask_num_orders8",
    "ask_num_orders9",
    "ask_num_orders10",
    "num_trades",
    "volume",
    "amount",
    "total_bid_volume",
    "total_ask_volume",
    "bid_avg_price",
    "ask_avg_price",
]

FIN_COLUMNS: list[str] = [
    "operating_revenue",
    "net_profit_to_parent_shareholders",
    "total_assets",
    "total_equity_to_parent_shareholders",
]


def resolve_datasources(datasources: dict[str, str] | None = None) -> dict[str, str]:
    out = dict(DEFAULT_DATASOURCES)
    if datasources:
        out.update(datasources)
    return out


def _date_filters(start: str, end: str) -> dict[str, list[Any]]:
    return {"date": [start, end]}


# ---------------------------------------------------------------------------
# Universe: historical CSI1000 (point-in-time via instruments table)
# ---------------------------------------------------------------------------

def fetch_csi1000_instruments(
    start: str = TRAIN_START,
    end: str = TRAIN_END,
    datasources: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return ``[date, instrument]`` membership for CSI1000 over ``[start, end]``.

    Uses ``bigalpha_2026_instruments`` (platform table; no need to remap).
    """
    ds = resolve_datasources(datasources)
    table = ds["instruments"]
    sql = f"""
    SELECT date, instrument
    FROM {table}
    ORDER BY date, instrument
    """
    df = query_df(sql, filters=_date_filters(start, end))
    if df is None or df.empty:
        raise RuntimeError(
            f"Empty universe from {table} between {start} and {end}. "
            "Check permission / date filters."
        )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype("string")
    return df.reset_index(drop=True)


def list_instruments_union(
    start: str = TRAIN_START,
    end: str = TRAIN_END,
    datasources: dict[str, str] | None = None,
) -> list[str]:
    """Distinct instruments that appear in the universe at any date in range."""
    df = fetch_csi1000_instruments(start, end, datasources)
    return sorted(df["instrument"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# 1-minute K + order book
# ---------------------------------------------------------------------------

def fetch_bar1m(
    start: str,
    end: str,
    instruments: list[str] | None = None,
    columns: list[str] | None = None,
    datasources: dict[str, str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load 1-minute bars + 10-level LOB from ``bigalpha_2026_stock_bar1m``.

    Example (official)::

        dai.query(
            "SELECT * FROM bigalpha_2026_stock_bar1m",
            filters={"date": ["2019-01-01 00:00:00", "2019-01-05 23:59:59"]},
        ).df().head()
    """
    ensure_login()
    ds = resolve_datasources(datasources)
    table = ds["bar1m"]
    cols = columns or BAR1M_COLUMNS
    select = ", ".join(cols)

    where_extra = ""
    filters: dict[str, list[Any]] = _date_filters(start, end)
    if instruments:
        # Prefer filters for partition pruning when supported; also keep SQL IN.
        in_list = ", ".join(f"'{x}'" for x in instruments)
        where_extra = f" WHERE instrument IN ({in_list})"
        filters["instrument"] = list(instruments)

    limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
    sql = f"SELECT {select} FROM {table}{where_extra}{limit_sql}"
    df = query_df(sql, filters=filters)
    if df is None or df.empty:
        raise RuntimeError(
            f"Empty bar1m from {table} between {start} and {end}. "
            "Check permission / filters."
        )
    return _normalize_bar1m(df)


def _normalize_bar1m(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    if "instrument" in out.columns:
        out["instrument"] = out["instrument"].astype("string")
    # If table exposes ``close`` instead of / in addition to ``price``, keep both.
    if "close" not in out.columns and "price" in out.columns:
        out["close"] = out["price"]
    return out


def aggregate_bar1m_to_daily(bar1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars to daily OHLCV for AlphaForge day-freq StockData.

    Notes on volume/amount: schema marks them as *当日累计*, so daily uses MAX
    rather than SUM. Open/high/low/close use first/max/min/last within the day.
    """
    df = bar1m.copy()
    if "close" not in df.columns and "price" in df.columns:
        df["close"] = df["price"]
    df = df.sort_values(["instrument", "date"])
    df["trading_day"] = pd.to_datetime(df["date"]).dt.normalize()

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "open": g["open"].iloc[0],
                "high": g["high"].max(),
                "low": g["low"].min(),
                "close": g["close"].iloc[-1],
                "volume": g["volume"].max(),
                "amount": g["amount"].max(),
            }
        )

    daily = (
        df.groupby(["trading_day", "instrument"], group_keys=False)
        .apply(_agg)
        .reset_index()
        .rename(columns={"trading_day": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["instrument"] = daily["instrument"].astype("string")
    return daily


def fetch_bar1m_as_daily(
    start: str,
    end: str,
    instruments: list[str] | None = None,
    datasources: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convenience: pull bar1m then aggregate to daily OHLCV."""
    # Only pull columns needed for daily OHLC aggregation.
    cols = [
        "date",
        "instrument",
        "open",
        "high",
        "low",
        "price",
        "volume",
        "amount",
    ]
    bar = fetch_bar1m(
        start, end, instruments=instruments, columns=cols, datasources=datasources
    )
    return aggregate_bar1m_to_daily(bar)


# ---------------------------------------------------------------------------
# PIT financials
# ---------------------------------------------------------------------------

def fetch_financial_pit(
    start: str,
    end: str,
    fin_cols: list[str] | None = None,
    datasources: dict[str, str] | None = None,
    category: str = "lf",
    shift: int = 0,
) -> pd.DataFrame:
    """Load PIT financials from ``bigalpha_2026_financial``.

    Default filter matches platform examples: ``category='lf' AND shift=0``.
    """
    ds = resolve_datasources(datasources)
    table = ds["financial"]
    cols = fin_cols or FIN_COLUMNS
    select = ", ".join(["date", "instrument", *cols])
    sql = f"""
    SELECT {select}
    FROM {table}
    WHERE category = '{category}' AND shift = {int(shift)}
    """
    df = query_df(sql, filters=_date_filters(start, end))
    if df is None or df.empty:
        raise RuntimeError(
            f"Empty financial from {table} between {start} and {end}. "
            "Check permission / category/shift."
        )
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["instrument"] = out["instrument"].astype("string")
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)


def ffill_financial_to_calendar(
    fin: pd.DataFrame,
    calendar_start: str,
    calendar_end: str,
    fin_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Map announcement-date financials onto a natural-day calendar with ffill."""
    cols = fin_cols or [c for c in FIN_COLUMNS if c in fin.columns]
    natural_dates = pd.date_range(
        start=pd.to_datetime(calendar_start).normalize(),
        end=pd.to_datetime(calendar_end).normalize(),
        freq="D",
    )
    base = fin.set_index("date").sort_index()

    def _one(group: pd.DataFrame) -> pd.DataFrame:
        name = group.name
        g = group.reindex(natural_dates)
        g[cols] = g[cols].ffill()
        g["instrument"] = name
        return g

    filled = base.groupby("instrument", group_keys=False).apply(_one)
    filled = (
        filled.reset_index()
        .rename(columns={"index": "date"})
        .dropna(subset=["instrument"])
    )
    filled["date"] = pd.to_datetime(filled["date"])
    filled["instrument"] = filled["instrument"].astype("string")
    return filled.reset_index(drop=True)
