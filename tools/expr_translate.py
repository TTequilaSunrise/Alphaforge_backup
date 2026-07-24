"""
AlphaForge / AlphaGen 公式字符串 → GP DSL / DAI SQL 双向翻译工具。

AlphaForge 公式示例::
    ts_corr(volume, Inv((1.0*vwap)), 30)
    (ts_var((-30.0-ts_corr(...)),50)--5.0)

约定:
    - AlphaForge 中 ``(a--b)`` 表示 ``a - (-b)``，即 ``a + b``（当 b 为正数）
    - GP DSL 使用 ``col/add/sub/mul/div/inv/s_log1p/ref/ts_*`` 等函数式写法
    - DAI 使用 ``m_lag/m_avg/m_corr`` 等面板 SQL 函数
"""

import csv
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

FEATURES: set[str] = {'open', 'high', 'low', 'close', 'volume', 'vwap', 'open_'}

ROLLING_UNARY: set[str] = {
    'Ref', 'ts_mean', 'ts_sum', 'ts_std', 'ts_var', 'ts_max', 'ts_min',
    'ts_med', 'ts_mad', 'ts_rank', 'ts_delta', 'ts_div', 'ts_pctchange',
    'ts_wma', 'ts_ema', 'ts_ir', 'ts_min_max_diff', 'ts_max_diff', 'ts_min_diff',
    'ts_skew', 'ts_kurt',
}

ROLLING_PAIR: set[str] = {'ts_cov', 'ts_corr'}

UNARY_OPS: set[str] = {'Inv', 'S_log1p', 'Abs', 'Sign', 'Log', 'CSRank'}


class TokKind(Enum):
    NUM = auto()
    IDENT = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    STARSTAR = auto()
    EOF = auto()


@dataclass
class Token:
    kind: TokKind
    value: str
    pos: int


@dataclass
class Num:
    value: float


@dataclass
class Feat:
    name: str


@dataclass
class Call:
    name: str
    args: list[object]


@dataclass
class Bin:
    op: str
    left: object
    right: object


@dataclass
class Neg:
    arg: object


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(src)

    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '(':
            tokens.append(Token(TokKind.LPAREN, ch, i))
            i += 1
            continue
        if ch == ')':
            tokens.append(Token(TokKind.RPAREN, ch, i))
            i += 1
            continue
        if ch == ',':
            tokens.append(Token(TokKind.COMMA, ch, i))
            i += 1
            continue
        if ch == '+':
            tokens.append(Token(TokKind.PLUS, ch, i))
            i += 1
            continue
        if ch == '*':
            if i + 1 < n and src[i + 1] == '*':
                tokens.append(Token(TokKind.STARSTAR, '**', i))
                i += 2
            else:
                tokens.append(Token(TokKind.STAR, ch, i))
                i += 1
            continue
        if ch == '/':
            tokens.append(Token(TokKind.SLASH, ch, i))
            i += 1
            continue
        if ch == '-':
            tokens.append(Token(TokKind.MINUS, ch, i))
            i += 1
            continue
        if ch.isdigit() or (ch == '.' and i + 1 < n and src[i + 1].isdigit()):
            start = i
            i += 1
            while i < n and (src[i].isdigit() or src[i] == '.'):
                i += 1
            tokens.append(Token(TokKind.NUM, src[start:i], start))
            continue
        if ch.isalpha() or ch == '_':
            start = i
            i += 1
            while i < n and (src[i].isalnum() or src[i] == '_'):
                i += 1
            tokens.append(Token(TokKind.IDENT, src[start:i], start))
            continue
        raise ValueError(f'无法解析字符 {ch!r}，位置 {i}，公式: {src!r}')

    tokens.append(Token(TokKind.EOF, '', n))
    return tokens


class _Parser:
    def __init__(self, src: str) -> None:
        self._src = src.strip()
        self._tokens = tokenize(self._src)
        self._pos = 0

    def parse(self) -> object:
        node = self._parse_expr()
        if self._cur().kind != TokKind.EOF:
            raise ValueError(f'公式尾部有多余内容: {self._src[self._cur().pos:]!r}')
        return node

    def _cur(self) -> Token:
        return self._tokens[self._pos]

    def _eat(self, kind: TokKind) -> Token:
        tok = self._cur()
        if tok.kind != kind:
            raise ValueError(
                f'期望 {kind.name}，得到 {tok.kind.name} ({tok.value!r})，'
                f'位置 {tok.pos}，公式: {self._src!r}'
            )
        self._pos += 1
        return tok

    def _parse_expr(self) -> object:
        return self._parse_add()

    def _parse_add(self) -> object:
        left = self._parse_mul()
        while True:
            tok = self._cur()
            if tok.kind == TokKind.PLUS:
                self._eat(TokKind.PLUS)
                left = Bin('add', left, self._parse_mul())
            elif tok.kind == TokKind.MINUS:
                if (
                    self._pos + 2 < len(self._tokens)
                    and self._tokens[self._pos + 1].kind == TokKind.MINUS
                    and self._tokens[self._pos + 2].kind == TokKind.NUM
                ):
                    self._eat(TokKind.MINUS)
                    self._eat(TokKind.MINUS)
                    num = float(self._eat(TokKind.NUM).value)
                    left = Bin('add', left, Num(num))
                else:
                    self._eat(TokKind.MINUS)
                    left = Bin('sub', left, self._parse_mul())
            else:
                break
        return left

    def _parse_mul(self) -> object:
        left = self._parse_unary()
        while self._cur().kind in (TokKind.STAR, TokKind.SLASH, TokKind.STARSTAR):
            if self._cur().kind == TokKind.STAR:
                self._eat(TokKind.STAR)
                left = Bin('mul', left, self._parse_unary())
            elif self._cur().kind == TokKind.SLASH:
                self._eat(TokKind.SLASH)
                left = Bin('div', left, self._parse_unary())
            else:
                self._eat(TokKind.STARSTAR)
                left = Bin('pow', left, self._parse_unary())
        return left

    def _parse_unary(self) -> object:
        if self._cur().kind == TokKind.MINUS:
            self._eat(TokKind.MINUS)
            return Neg(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> object:
        tok = self._cur()
        if tok.kind == TokKind.NUM:
            self._eat(TokKind.NUM)
            return Num(float(tok.value))
        if tok.kind == TokKind.IDENT:
            name = tok.value
            self._eat(TokKind.IDENT)
            if self._cur().kind == TokKind.LPAREN:
                self._eat(TokKind.LPAREN)
                args: list[object] = []
                if self._cur().kind != TokKind.RPAREN:
                    args.append(self._parse_expr())
                    while self._cur().kind == TokKind.COMMA:
                        self._eat(TokKind.COMMA)
                        args.append(self._parse_expr())
                self._eat(TokKind.RPAREN)
                return Call(name, args)
            if name not in FEATURES:
                raise ValueError(f'未知字段或函数: {name!r}，公式: {self._src!r}')
            return Feat(name)
        if tok.kind == TokKind.LPAREN:
            self._eat(TokKind.LPAREN)
            node = self._parse_expr()
            self._eat(TokKind.RPAREN)
            return node
        raise ValueError(f'意外的 token {tok.kind.name}，位置 {tok.pos}，公式: {self._src!r}')


def parse_alphaforge_expr(src: str) -> object:
    return _Parser(src).parse()


def _is_simple_feat(node: object) -> str | None:
    if isinstance(node, Feat):
        return node.name
    return None


def _fmt_num(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return repr(v)


def emit_gp(node: object) -> str:
    if isinstance(node, Num):
        return _fmt_num(node.value)
    if isinstance(node, Feat):
        return f"col('{node.name}')"
    if isinstance(node, Neg):
        inner = emit_gp(node.arg)
        if isinstance(node.arg, Num):
            return _fmt_num(-node.arg.value)
        return f"neg({inner})"
    if isinstance(node, Bin):
        op_map = {
            'add': 'add',
            'sub': 'sub',
            'mul': 'mul',
            'div': 'div',
            'pow': 'pow',
        }
        fn = op_map[node.op]
        return f"{fn}({emit_gp(node.left)}, {emit_gp(node.right)})"
    if isinstance(node, Call):
        return _emit_gp_call(node)
    raise TypeError(f'未知 AST 节点: {node!r}')


def _emit_gp_call(node: Call) -> str:
    name = node.name
    args = node.args

    if name == 'Inv':
        return f"inv({emit_gp(args[0])})"
    if name == 'S_log1p':
        return f"s_log1p({emit_gp(args[0])})"
    if name == 'Abs':
        return f"abs_({emit_gp(args[0])})"
    if name == 'Sign':
        return f"sign({emit_gp(args[0])})"
    if name == 'Log':
        return f"log_({emit_gp(args[0])})"
    if name == 'CSRank':
        return f"rank({emit_gp(args[0])})"

    if name == 'Ref':
        feat = _is_simple_feat(args[0])
        if feat is not None:
            return f"ts_delay('{feat}', {emit_gp(args[1])})"
        return f"ref({emit_gp(args[0])}, {emit_gp(args[1])})"

    gp_roll_map: dict[str, str] = {
        'ts_mean': 'ts_mean_s',
        'ts_sum': 'ts_sum_s',
        'ts_std': 'ts_std_s',
        'ts_var': 'ts_var',
        'ts_max': 'ts_max_s',
        'ts_min': 'ts_min_s',
        'ts_med': 'ts_med_s',
        'ts_mad': 'ts_mad_s',
        'ts_rank': 'ts_rank_s',
        'ts_delta': 'delta',
        'ts_pctchange': 'ts_pct_s',
        'ts_wma': 'ts_wma_s',
        'ts_ema': 'ts_ema_s',
        'ts_div': 'ts_div_s',
        'ts_corr': 'ts_corr',
        'ts_cov': 'ts_cov',
    }

    if name in gp_roll_map:
        fn = gp_roll_map[name]
        if name in ROLLING_PAIR:
            return f"{fn}({emit_gp(args[0])}, {emit_gp(args[1])}, {emit_gp(args[2])})"
        if name == 'ts_delta':
            return f"delta({emit_gp(args[0])}, {emit_gp(args[1])})"
        feat = _is_simple_feat(args[0])
        if feat is not None and fn.endswith('_s') is False and name not in ('ts_var',):
            legacy: dict[str, str] = {
                'ts_mean': 'ts_mean',
                'ts_sum': 'ts_sum',
                'ts_std': 'ts_std',
                'ts_max': 'ts_max',
                'ts_min': 'ts_min',
            }
            if name in legacy:
                return f"{legacy[name]}('{feat}', {emit_gp(args[1])})"
        return f"{fn}({emit_gp(args[0])}, {emit_gp(args[1])})"

    raise ValueError(f'GP 翻译不支持算子: {name}')


def emit_dai(node: object) -> str:
    if isinstance(node, Num):
        return _fmt_num(node.value)
    if isinstance(node, Feat):
        col = 'open' if node.name == 'open_' else node.name
        return col
    if isinstance(node, Neg):
        if isinstance(node.arg, Num):
            return _fmt_num(-node.arg.value)
        return f"(-({emit_dai(node.arg)}))"
    if isinstance(node, Bin):
        op_char = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}[node.op]
        left = emit_dai(node.left)
        right = emit_dai(node.right)
        return f"({left} {op_char} {right})"
    if isinstance(node, Call):
        return _emit_dai_call(node)
    raise TypeError(f'未知 AST 节点: {node!r}')


def _emit_dai_call(node: Call) -> str:
    name = node.name
    args = node.args

    if name == 'Inv':
        return f"(1.0 / ({emit_dai(args[0])}))"
    if name == 'S_log1p':
        x = emit_dai(args[0])
        return f"(sign({x}) * log(1 + abs({x})))"
    if name == 'Abs':
        return f"abs({emit_dai(args[0])})"
    if name == 'Sign':
        return f"sign({emit_dai(args[0])})"
    if name == 'Log':
        return f"log({emit_dai(args[0])})"
    if name == 'CSRank':
        return f"c_pct_rank({emit_dai(args[0])})"

    if name == 'Ref':
        return f"m_lag({emit_dai(args[0])}, {emit_dai(args[1])})"

    dai_roll_map: dict[str, str] = {
        'ts_mean': 'm_avg',
        'ts_sum': 'm_sum',
        'ts_std': 'm_stddev',
        'ts_var': 'm_nanvar',
        'ts_max': 'm_max',
        'ts_min': 'm_min',
        'ts_med': 'm_median',
        'ts_mad': 'm_mad',
        'ts_rank': 'm_pct_rank',
        'ts_delta': 'm_delta',
        'ts_pctchange': 'ts_pctchange_dai',
        'ts_corr': 'm_corr',
        'ts_cov': 'm_covar_samp',
        'ts_wma': 'm_avg',
        'ts_ema': 'm_avg',
        'ts_div': 'm_avg',
    }

    if name in dai_roll_map:
        fn = dai_roll_map[name]
        if name == 'ts_pctchange':
            x = emit_dai(args[0])
            w = emit_dai(args[1])
            return f"(({x}) / m_lag({x}, {w}) - 1.0)"
        if name in ROLLING_PAIR:
            return f"{fn}({emit_dai(args[0])}, {emit_dai(args[1])}, {emit_dai(args[2])})"
        return f"{fn}({emit_dai(args[0])}, {emit_dai(args[1])})"

    raise ValueError(f'DAI 翻译不支持算子: {name}')


def translate_expr(alphaforge_expr: str) -> dict[str, str]:
    node = parse_alphaforge_expr(alphaforge_expr)
    return {
        'alphaforge': alphaforge_expr,
        'gp_dsl': emit_gp(node),
        'dai_sql': emit_dai(node),
    }


def translate_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> list[dict[str, str | float]]:
    input_path = Path(input_path)
    rows: list[dict[str, str | float]] = []
    with input_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'CSV 无表头: {input_path}')
        for row in reader:
            expr = row.get('exprs') or row.get('expr') or ''
            if not expr:
                continue
            tr = translate_expr(expr)
            out_row: dict[str, str | float] = {
                'exprs': expr,
                'gp_dsl': tr['gp_dsl'],
                'dai_sql': tr['dai_sql'],
            }
            if 'scores' in row and row['scores']:
                out_row['scores'] = float(row['scores'])
            elif 'score' in row and row['score']:
                out_row['scores'] = float(row['score'])
            rows.append(out_row)

    out_path = Path(output_path) if output_path else input_path.with_name(
        input_path.stem + '_translated.csv'
    )
    fieldnames = ['exprs', 'gp_dsl', 'dai_sql', 'scores']
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return rows


def translate_expr_list(exprs: list[str]) -> list[dict[str, str]]:
    return [translate_expr(e) for e in exprs]


def to_dai_sql(alphaforge_expr: str) -> str:
    return translate_expr(alphaforge_expr)['dai_sql']


def to_gp_dsl(alphaforge_expr: str) -> str:
    return translate_expr(alphaforge_expr)['gp_dsl']


def format_expr_output(alphaforge_expr: str, fmt: str) -> str:
    if fmt == 'gp':
        return to_gp_dsl(alphaforge_expr)
    if fmt == 'alphaforge':
        return alphaforge_expr
    return to_dai_sql(alphaforge_expr)
