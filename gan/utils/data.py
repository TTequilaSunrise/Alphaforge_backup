from alphagen_generic.features import *
from alphagen.data.expression import *

import os


def get_data_by_year(
    train_start=2019,
    train_end=2023,
    valid_year=2024,
    test_year=2024,
    instruments="csi1000",
    target=None,
    freq="day",
    backend: str = "bigalpha2026",
    datasources: dict | None = None,
):
    """Load train/valid/test StockData via BigQuant.

    Default backend ``bigalpha2026``:
    - universe from ``bigalpha_2026_instruments`` (CSI1000 PIT)
    - bars from ``bigalpha_2026_stock_bar1m`` aggregated to daily for AFF

    For official A-share daily tables, pass ``backend="cn_bar1d"``.
    """
    from gan.utils import load_pickle, save_pickle
    from alphagen_qlib.stock_data import StockData

    train_dates = (f"{train_start}-01-01", f"{train_end}-12-31")
    val_dates = (f"{valid_year}-01-01", f"{valid_year}-12-31")
    test_dates = (f"{test_year}-01-01", f"{test_year}-12-31")

    train_start_s, train_end_s = train_dates
    valid_start, valid_end = val_dates
    valid_head_start = f"{valid_year - 2}-01-01"
    test_start, test_end = test_dates
    test_head_start = f"{test_year - 2}-01-01"

    name = (
        f"bq_{backend}_{instruments}_pkl_"
        + str(target).replace("/", "_").replace(" ", "")
        + "_"
        + freq
    )
    name = (
        f"{name}_{train_start_s}_{train_end_s}_{valid_start}_{valid_end}_"
        f"{test_start}_{test_end}"
    )

    def _load(start: str, end: str):
        return StockData(
            instruments,
            start,
            end,
            raw=True,
            freq=freq,
            backend=backend,
            datasources=datasources,
        )

    try:
        data = load_pickle(f"pkl/{name}/data.pkl")
        data_valid = load_pickle(f"pkl/{name}/data_valid.pkl")
        data_valid_withhead = load_pickle(f"pkl/{name}/data_valid_withhead.pkl")
        data_test = load_pickle(f"pkl/{name}/data_test.pkl")
        data_test_withhead = load_pickle(f"pkl/{name}/data_test_withhead.pkl")
    except Exception:
        print(f"Data not exist, load from BigQuant backend={backend}")
        data = _load(train_start_s, train_end_s)
        data_valid = _load(valid_start, valid_end)
        data_valid_withhead = _load(valid_head_start, valid_end)
        data_test = _load(test_start, test_end)
        data_test_withhead = _load(test_head_start, test_end)

        os.makedirs(f"pkl/{name}", exist_ok=True)
        save_pickle(data, f"pkl/{name}/data.pkl")
        save_pickle(data_valid, f"pkl/{name}/data_valid.pkl")
        save_pickle(data_valid_withhead, f"pkl/{name}/data_valid_withhead.pkl")
        save_pickle(data_test, f"pkl/{name}/data_test.pkl")
        save_pickle(data_test_withhead, f"pkl/{name}/data_test_withhead.pkl")

    try:
        data_all = load_pickle(f"pkl/{name}/data_all.pkl")
    except Exception:
        data_all = _load(train_start_s, test_end)
        save_pickle(data_all, f"pkl/{name}/data_all.pkl")
    return (
        data_all,
        data,
        data_valid,
        data_valid_withhead,
        data_test,
        data_test_withhead,
        name,
    )
