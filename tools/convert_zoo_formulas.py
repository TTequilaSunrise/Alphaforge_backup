"""批量转换 AlphaForge 因子 zoo 中的公式为 GP DSL / DAI SQL。"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.expr_translate import translate_csv, translate_expr, translate_expr_list


def _find_csv_files(root: Path) -> list[Path]:
    return sorted(root.glob('**/csv_*.csv'))


def main() -> None:
    parser = argparse.ArgumentParser(description='转换 AlphaForge 公式为 GP DSL / DAI SQL')
    parser.add_argument(
        '--root',
        default=str(_ROOT / 'out'),
        help='搜索 csv_zoo / csv_* 的根目录',
    )
    parser.add_argument(
        '--expr',
        action='append',
        default=[],
        help='单条 AlphaForge 公式（可重复）',
    )
    args = parser.parse_args()

    if args.expr:
        rows = translate_expr_list(args.expr)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    root = Path(args.root)
    csv_files = _find_csv_files(root)
    if not csv_files:
        print(f'未找到 csv 文件: {root}')
        return

    all_rows: list[dict[str, str]] = []
    for csv_path in csv_files:
        rows = translate_csv(csv_path)
        out_path = csv_path.with_name(csv_path.stem + '_translated.csv')
        print(f'已转换: {csv_path} -> {out_path} ({len(rows)} 条)')
        all_rows.extend(rows)

    catalog_path = root / 'formulas_translated.json'
    catalog_path.write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'汇总: {catalog_path}')


if __name__ == '__main__':
    main()
