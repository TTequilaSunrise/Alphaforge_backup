"""扩展 GP DSL 求值：支持 AlphaForge 翻译后的 inv/ref/s_log1p/ts_var 等算子。"""

import sys
from pathlib import Path

_GP_ROOT = Path(__file__).resolve().parents[2] / 'Factor Mining'
if str(_GP_ROOT) not in sys.path:
    sys.path.insert(0, str(_GP_ROOT))


def eval_translated_formula(df: object, gp_expr: str) -> object:
    """求值 GP DSL 字符串（含 AlphaForge 翻译算子）。"""
    import numpy as np
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df 必须是 DataFrame')

    frame = df.sort_values(['instrument', 'date']).copy()
    inst = frame['instrument']
    inst_g = frame.groupby('instrument', group_keys=False)
    inst_labels = frame['instrument']

    def col(name: str) -> pd.Series:
        if name not in frame.columns:
            raise ValueError(f'未知字段: {name}')
        return frame[name]

    def _roll_min_periods(n: int, *, at_least: int = 1) -> int:
        return max(at_least, int(n) // 4)

    def _as_series(v: object) -> pd.Series:
        if isinstance(v, pd.Series):
            return v
        return pd.Series(float(v), index=frame.index, dtype=float)

    def _resolve(v: object) -> pd.Series:
        if isinstance(v, str):
            return col(v)
        return _as_series(v)

    def ts_pct(name: str, n: int) -> pd.Series:
        return inst_g[name].pct_change(int(n))

    def ts_std(name: str, n: int) -> pd.Series:
        return inst_g[name].transform(
            lambda s: s.rolling(int(n), min_periods=_roll_min_periods(n, at_least=2)).std()
        )

    def ts_mean(name: str, n: int) -> pd.Series:
        return inst_g[name].transform(
            lambda s: s.rolling(int(n), min_periods=_roll_min_periods(n)).mean()
        )

    def ts_sum(name: str, n: int) -> pd.Series:
        return inst_g[name].transform(
            lambda s: s.rolling(int(n), min_periods=_roll_min_periods(n)).sum()
        )

    def ts_max(name: str, n: int) -> pd.Series:
        return inst_g[name].transform(
            lambda s: s.rolling(int(n), min_periods=_roll_min_periods(n)).max()
        )

    def ts_min(name: str, n: int) -> pd.Series:
        return inst_g[name].transform(
            lambda s: s.rolling(int(n), min_periods=_roll_min_periods(n)).min()
        )

    def ts_delay(name: str, n: int) -> pd.Series:
        return inst_g[name].shift(int(n))

    def ts_rank(name: str, n: int) -> pd.Series:
        win = int(n)
        mp = _roll_min_periods(win, at_least=2)

        def _last_pct_rank(arr: object) -> float:
            a = np.asarray(arr, dtype=float)
            if a.size == 0 or a[-1] != a[-1]:
                return float('nan')
            valid = a[np.isfinite(a)]
            if valid.size == 0:
                return float('nan')
            return float(np.sum(valid <= a[-1]) / valid.size)

        return inst_g[name].transform(
            lambda s: s.rolling(win, min_periods=mp).apply(_last_pct_rank, raw=True)
        )

    def ts_zscore(name: str, n: int) -> pd.Series:
        win = int(n)
        mp = _roll_min_periods(win, at_least=2)

        def _z(s: pd.Series) -> pd.Series:
            m = s.rolling(win, min_periods=mp).mean()
            sd = s.rolling(win, min_periods=mp).std()
            return (s - m) / sd.replace(0, np.nan)

        return inst_g[name].transform(_z)

    def rank(x: object) -> pd.Series:
        s = pd.Series(x, index=frame.index, dtype=float)
        return s.groupby(frame['date']).rank(pct=True, method='average')

    def ts_corr(a: object, b: object, n: int) -> pd.Series:
        sa = _resolve(a)
        sb = _resolve(b)
        win = int(n)
        mp = _roll_min_periods(win, at_least=3)
        out = pd.Series(np.nan, index=frame.index, dtype=float)
        for _, idx in frame.groupby('instrument').groups.items():
            ca = sa.loc[idx].rolling(win, min_periods=mp).corr(sb.loc[idx])
            out.loc[idx] = ca.to_numpy()
        return out

    def delta(x: object, n: int) -> pd.Series:
        s = _resolve(x)
        return s.groupby(inst_labels, group_keys=False).diff(int(n))

    def add(a: object, b: object) -> pd.Series:
        return _as_series(a) + _as_series(b)

    def sub(a: object, b: object) -> pd.Series:
        return _as_series(a) - _as_series(b)

    def mul(a: object, b: object) -> pd.Series:
        return _as_series(a) * _as_series(b)

    def div(a: object, b: object) -> pd.Series:
        return _as_series(a) / _as_series(b).replace(0, np.nan)

    def neg(x: object) -> pd.Series:
        return -_as_series(x)

    def abs_(x: object) -> pd.Series:
        return _as_series(x).abs()

    def sign(x: object) -> pd.Series:
        return np.sign(_as_series(x))

    def log_(x: object) -> pd.Series:
        return np.log(_as_series(x).abs().clip(lower=1e-12))

    def pow(a: object, b: object) -> pd.Series:
        return _as_series(a) ** _as_series(b)

    def inv(x: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        return (1.0 / s.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    def s_log1p(x: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        return np.sign(s) * np.log1p(s.abs())

    def ref(x: object, n: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        return s.groupby(inst, group_keys=False).shift(int(n))

    def ts_var(x: object, n: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        win = int(n)
        mp = _roll_min_periods(win, at_least=2)
        return (
            s.groupby(inst, group_keys=False)
            .transform(lambda v: v.rolling(win, min_periods=mp).var())
            .replace([np.inf, -np.inf], np.nan)
        )

    def _rolling_apply(s: pd.Series, win: int, fn: object) -> pd.Series:
        mp = _roll_min_periods(win, at_least=1)
        return s.groupby(inst, group_keys=False).transform(
            lambda v: fn(v.rolling(win, min_periods=mp))
        )

    def ts_sum_s(x: object, n: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        return _rolling_apply(s, int(n), lambda r: r.sum())

    def ts_mean_s(x: object, n: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        return _rolling_apply(s, int(n), lambda r: r.mean())

    def ts_std_s(x: object, n: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        return _rolling_apply(s, int(n), lambda r: r.std())

    def ts_max_s(x: object, n: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        return _rolling_apply(s, int(n), lambda r: r.max())

    def ts_min_s(x: object, n: object) -> pd.Series:
        s = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        return _rolling_apply(s, int(n), lambda r: r.min())

    def ts_cov(x: object, y: object, n: object) -> pd.Series:
        sa = _resolve(x) if not isinstance(x, pd.Series) else _as_series(x)
        sb = _resolve(y) if not isinstance(y, pd.Series) else _as_series(y)
        win = int(n)
        mp = _roll_min_periods(win, at_least=3)
        out = pd.Series(np.nan, index=frame.index, dtype=float)
        for _, idx in frame.groupby('instrument').groups.items():
            out.loc[idx] = (
                sa.loc[idx].rolling(win, min_periods=mp).cov(sb.loc[idx]).to_numpy()
            )
        return out.replace([np.inf, -np.inf], np.nan)

    safe_globals: dict[str, object] = {'__builtins__': {}}
    safe_locals: dict[str, object] = {
        'col': col,
        'ts_pct': ts_pct,
        'ts_std': ts_std,
        'ts_mean': ts_mean,
        'ts_sum': ts_sum,
        'ts_max': ts_max,
        'ts_min': ts_min,
        'ts_delay': ts_delay,
        'ts_rank': ts_rank,
        'ts_zscore': ts_zscore,
        'ts_corr': ts_corr,
        'corr': ts_corr,
        'delta': delta,
        'rank': rank,
        'add': add,
        'sub': sub,
        'mul': mul,
        'div': div,
        'neg': neg,
        'abs_': abs_,
        'sign': sign,
        'log_': log_,
        'pow': pow,
        'inv': inv,
        's_log1p': s_log1p,
        'ref': ref,
        'ts_var': ts_var,
        'ts_sum_s': ts_sum_s,
        'ts_mean_s': ts_mean_s,
        'ts_std_s': ts_std_s,
        'ts_max_s': ts_max_s,
        'ts_min_s': ts_min_s,
        'ts_cov': ts_cov,
    }

    out = eval(gp_expr, safe_globals, safe_locals)
    factor = pd.Series(out, index=frame.index, dtype=float)
    return factor.replace([np.inf, -np.inf], np.nan)
