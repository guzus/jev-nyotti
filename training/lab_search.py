"""Evaluate decision-rule configs on the TUNE window (2023) and append results to a benchmark run file.

  python training/lab_search.py --family turnover --configs configs.json   # list of rule dicts
Each line of benchmarks/runs/<family>.jsonl: {family, rules, window, fill, mean, per_symbol}. The
SELECT (2024) window is scored centrally by training/lab_select.py; confirmation data is not on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnl_lab  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import os
MODEL = Path(os.environ.get('LAB_MODEL', ROOT / '.runtime' / 'numeric-policy-v4.json'))
SHA = os.environ.get('LAB_SHA', '9101f3f5d92418f0de055962354c729c47c05c3b864ea9288bbccfc666992392')
TUNE = ROOT / '.runtime' / 'lab-2023'
SYMBOLS = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--family', required=True)
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--fill', choices=('close', 'next_open'), default='close')
    args = parser.parse_args()
    model = pnl_lab.numeric_policy.load(MODEL, SHA)
    cache: dict = {}
    out = ROOT / 'benchmarks' / 'runs' / f'{args.family}.jsonl'
    for rules in json.loads(args.configs.read_text()):
        result = pnl_lab.evaluate(model, rules, TUNE, SYMBOLS, args.fill, cache)
        row = dict(family=args.family, model_sha=SHA, rules=rules, window=result['window'], fill=args.fill,
                   mean={k: round(v, 4) for k, v in result['mean'].items()},
                   per_symbol={k: {q: round(v[q], 3) for q in ('net_return_pct', 'fees_pct', 'trades', 'max_drawdown_pct')}
                               for k, v in result['per_symbol'].items()})
        with out.open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(dict(rules=rules, net=row['mean']['net_return_pct'], fees=row['mean']['fees_pct'],
                              trades=row['mean']['trades'], sharpe=row['mean']['daily_sharpe'])), flush=True)


if __name__ == '__main__':
    main()
