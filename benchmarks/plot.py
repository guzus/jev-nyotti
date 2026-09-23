"""Render the autoresearch-style progress chart from benchmarks/runs/*.jsonl and benchmarks/versions.json.

  .runtime/action-venv/bin/python benchmarks/plot.py  ->  benchmarks/progress.png
Top: every tune-window experiment in evaluation order (grey), running best (green step line).
Bottom: shipped versions on the fixed benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = {'turnover': '#2a6f97', 'confidence': '#c9184a', 'trend': '#6a994e', 'combo': '#e09f3e'}


def runs() -> list[dict]:
    rows = []
    for path in sorted((HERE / 'runs').glob('*.jsonl')):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def main() -> None:
    rows = runs()
    versions = json.loads((HERE / 'versions.json').read_text())
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={'height_ratios': [3, 2]})
    best, xs, ys = float('-inf'), [], []
    for i, row in enumerate(rows):
        y = row['mean']['net_return_pct']
        top.scatter(i, y, s=10, color=COLORS.get(row['family'], '#999'), alpha=.45, linewidths=0)
        if y > best:
            best = y
            xs.append(i)
            ys.append(y)
    if xs:
        top.step(xs + [len(rows) - 1], ys + [ys[-1]], where='post', color='#1b9e4b', lw=2, label='running best')
        top.scatter(xs, ys, color='#1b9e4b', s=28, zorder=3)
    for family, color in COLORS.items():
        top.scatter([], [], color=color, label=family)
    top.axhline(0, color='#333', lw=.8, ls=':')
    top.axhline(311.1, color='#555', lw=1, ls='--')
    top.annotate('2023 buy & hold (+311 %): in-sample bull year, not skill', (0, 311.1), xytext=(4, 4), textcoords='offset points', fontsize=8, color='#555')
    top.set_yscale('symlog', linthresh=10)
    top.set_title(f'V5 decision-rule search · IN-SAMPLE 2023 tune window · {len(rows)} experiments · fee-inclusive 4-coin mean net %')
    top.set_xlabel('experiment # (grouped by search family)')
    top.set_ylabel('net return % (symlog)')
    top.legend(loc='lower right', fontsize=8, ncol=5)
    names = [v['version'] for v in versions]
    values = [v['benchmark_net_pct'] for v in versions]
    bars = bottom.bar(names, values, color=['#1b9e4b' if v > 0 else '#c0392b' for v in values])
    for bar, v in zip(bars, versions):
        bottom.annotate(v.get('note', ''), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center',
                        va='bottom' if bar.get_height() >= 0 else 'top', fontsize=7)
    bottom.axhline(0, color='#333', lw=.8)
    bottom.set_yscale('symlog', linthresh=10)
    bottom.set_title(versions[0].get('benchmark', 'benchmark'))
    bottom.set_ylabel('net return % (symlog)')
    fig.tight_layout()
    fig.savefig(HERE / 'progress.png', dpi=130)
    print(HERE / 'progress.png')


if __name__ == '__main__':
    main()
