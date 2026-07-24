# AlphaForge(AFF) — BigAlpha 2026 / BigQuant DAI


### Data contract (training set)

| Item | Spec |
|------|------|
| Universe | CSI1000 historical constituents (`bigalpha_2026_instruments`) |
| Range | 2019-01-01 ~ 2024-12-31 |
| Bars | 1-minute K + 10-level LOB (`bigalpha_2026_stock_bar1m`) |
| Financial | PIT (`bigalpha_2026_financial`, `category='lf' AND shift=0`) |

Official read example:

```python
import dai  # on platform; locally: bigquant.init_from_config(); from bigquant import dai

dai.query(
    "SELECT * FROM bigalpha_2026_stock_bar1m",
    filters={"date": ["2019-01-01 00:00:00", "2019-01-05 23:59:59"]},
).df().head()
```


### Code layout

| Module | Role |
|--------|------|
| [`alphagen_qlib/bigalpha2026.py`](alphagen_qlib/bigalpha2026.py) | Competition tables: instruments / bar1m / financial PIT / 1m→daily agg |
| [`alphagen_qlib/bigquant_backend.py`](alphagen_qlib/bigquant_backend.py) | Auth + generic `dai.query`; optional `cn_stock_bar1d` path |
| [`alphagen_qlib/stock_data.py`](alphagen_qlib/stock_data.py) | `StockData(..., backend="bigalpha2026")` day tensor for AFF |
| [`gan/utils/data.py`](gan/utils/data.py) | `get_data_by_year` defaults to csi1000 + bigalpha2026 |


### Env

```bash
conda activate bigquant
bq auth configure
# tables need ACL: bigalpha_2026_stock_bar1m / financial / instruments
```

Raw minute+LOB (debug):

```python
from alphagen_qlib.bigalpha2026 import fetch_bar1m, fetch_financial_pit, fetch_csi1000_instruments

pool = fetch_csi1000_instruments("2019-01-01", "2019-01-05")
bars = fetch_bar1m("2019-01-01 00:00:00", "2019-01-05 23:59:59")
fin = fetch_financial_pit("2018-01-01", "2019-01-05")
```

Day-freq StockData for AFF (aggregates 1m→1d):

```python
from alphagen_qlib.stock_data import StockData
import torch
d = StockData(
    "csi1000", "2019-01-01", "2019-03-31",
    raw=True, device=torch.device("cpu"),
    backend="bigalpha2026",
    max_backtrack_days=5, max_future_days=2,
)
print(d.data.shape)
```

Fallback to official daily table: `backend="cn_bar1d"`.


### Run AFF

```shell
python train_AFF.py --instruments=csi1000 --train_end_year=2023 --seeds=[0] --save_name=test --zoo_size=100
```

(Adjust years to fit 2019–2024; first run caches under `pkl/bq_bigalpha2026_*`.)


### Notes

- Schema has ``price`` (last) not ``close``; loader maps ``close = price`` when needed.
- ``volume`` / ``amount`` are documented as *当日累计*; daily aggregation uses ``MAX``, not ``SUM``.
- Platform may inject physical table names via ``datasources={...}``; pass through to `StockData` / `fetch_*`.
