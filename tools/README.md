# AlphaForge 公式翻译

## 默认行为（无需单独转换）

从 `alphagen/config.py` 起，因子生成时 **`exprs_str` 与 CSV 的 `exprs` 列默认为 DAI SQL**：

```python
EXPR_OUTPUT_FORMAT = 'dai'   # 可选: 'dai' | 'gp' | 'alphaforge'
```

`train_AFF.py` 保存 zoo 时输出：

| 列 | 含义 |
|----|------|
| `exprs` | DAI SQL（可直接用于 BigQuant） |
| `exprs_af` | AlphaForge 原始公式（内部对照） |
| `gp_dsl` | Factor Mining GP DSL |

Stage 2 `combine_AFF.py` 仍从 pickle 里的 **Expression 对象** 计算，不受字符串格式影响。

## 手动转换（仅旧 CSV 需要）

若 CSV 里 `exprs` 仍是 AlphaForge 格式，可运行：

```bash
python tools/convert_zoo_formulas.py
```

## GP 本地求值

```python
from tools.gp_eval import eval_translated_formula
factor = eval_translated_formula(panel_df, gp_dsl_string)
```

## 算子对照

| AlphaForge | GP DSL | DAI SQL |
|------------|--------|---------|
| `close` | `col('close')` | `close` |
| `Inv(x)` | `inv(x)` | `(1.0 / (x))` |
| `Ref(x,n)` | `ts_delay('x',n)` / `ref(expr,n)` | `m_lag(x,n)` |
| `ts_corr(a,b,n)` | `ts_corr(a,b,n)` | `m_corr(a,b,n)` |
| `(expr--5.0)` | `add(expr, 5)` | `(expr + 5)` |
