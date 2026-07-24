from alphagen_qlib.stock_data import StockData
from alphagen_qlib.bigquant_backend import resolve_instruments


def get_data_my(instru, start, end, raw=False, qlib_path="", freq="day"):
    """Load StockData from BigQuant. qlib_path is ignored (API compatibility)."""
    _ = qlib_path
    if isinstance(instru, str) and instru.lower() in {
        "csi300",
        "csi500",
        "csi800",
        "csi1000",
    }:
        # Keep universe name so StockData can resolve constituents itself,
        # but also allow callers that already expanded the list.
        return StockData(instru, start, end, raw=raw, freq=freq)

    if isinstance(instru, str):
        codes = resolve_instruments(instru, start, end)
    else:
        codes = resolve_instruments(list(instru), start, end)
    return StockData(codes, start, end, raw=raw, freq=freq)
