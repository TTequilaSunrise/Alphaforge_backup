from alphagen.data.expression import *


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

OPERATORS = Operators
OPERATORS = [
    # Unary
    # Abs,  
    # Sign,
    # Log,
    Inv,
    S_log1p,
    # CSRank,

    # Binary,
    Add, Sub, Mul, Div, 
    Pow,
    # Greater, Less,

    # Rolling
    Ref, ts_mean, ts_sum, ts_std, ts_var,  
    # ts_skew, 
    # ts_kurt,
    ts_max, ts_min,
    ts_med, ts_mad,  
    # ts_rank,

    ts_div,
    ts_pctchange,
    # ts_ir,
    # ts_min_max_diff,
    # ts_max_diff,ts_min_diff,
    ts_delta, ts_wma, ts_ema,

    # Pair rolling
    ts_cov, ts_corr
]

DELTA_TIMES = [1,5,10, 20, 30, 40, 50]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.

# 因子公式对外字符串格式（build_exprs / CSV 的 exprs 列）:
#   'dai'         -> BigQuant DAI SQL（默认）
#   'gp'          -> Factor Mining GP DSL
#   'alphaforge'  -> AlphaForge 原始格式
EXPR_OUTPUT_FORMAT = 'dai'
