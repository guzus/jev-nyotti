"""Causal, identifier-free BTC exposure imitation dataset from supplied execution CSVs.

No raw records, wallet fields, target-window fills or future candles enter model inputs.
The resulting task is historical position-side imitation, not profitable trade advice.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import pstdev

HOUR = 3600
LOOKBACK = 96
ACTIONS = ("long", "short", "flat")
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
INSTRUCTIONS = (
    "Imitate the supplied historical trader's position side at the end of the next hour "
    "using only this closed-candle snapshot and the position side immediately before the cutoff. "
    "Long and short describe signed XBTUSD contract exposure; flat means zero contracts. "
    "This is historical behavior prediction, not price direction, a recommendation or an order. "
    "Do not infer missing balances, leverage, news or future executions."
)


def timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Export is timezone-naive. UTC is an explicit documented pilot assumption.
    return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()


def iso(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def side(quantity: int) -> str:
    return "long" if quantity > 0 else "short" if quantity < 0 else "flat"


def read_positions(paths: list[Path]):
    events, seen, counts = [], set(), Counter()
    for path in sorted(paths):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["symbol"] != "XBTUSD":
                    continue
                kind = row["exectype"]
                if kind not in ("Trade", "Settlement", "Funding"):
                    raise ValueError(f"unsupported BTC event type: {kind}")
                identifier = row["execid"]
                if not identifier or identifier in seen:
                    raise ValueError("missing or duplicate execution identifier")
                seen.add(identifier)
                raw_qty = float(row["lastqty"])
                if not math.isfinite(raw_qty) or raw_qty < 0 or not raw_qty.is_integer():
                    raise ValueError("invalid integral contract quantity")
                qty = int(raw_qty)
                if kind != "Funding" and (qty <= 0 or row["side"] not in ("Buy", "Sell")):
                    raise ValueError("invalid position-changing event")
                events.append((timestamp(row["transacttime"]), 0 if kind == "Funding" else 1,
                               qty if kind == "Funding" or row["side"] == "Buy" else -qty))
                counts[kind] += 1
    events.sort()  # Funding checks the pre-trade position at a shared timestamp.
    position, anchors = 0, 0
    times, positions = [], []
    for time, priority, qty in events:
        if priority == 0:
            if abs(position) != qty:
                raise ValueError(f"position does not reconcile with funding at {iso(time)}")
            anchors += 1
        else:
            position += qty
            if times and times[-1] == time:
                positions[-1] = position
            else:
                times.append(time)
                positions.append(position)
    if anchors < 100 or not times:
        raise ValueError("insufficient position reconciliation evidence")
    return times, positions, {"events": dict(counts), "funding_anchors_matched": anchors,
                              "first_execution": iso(times[0]), "last_execution": iso(times[-1])}


def position_before(times: list[float], positions: list[int], cutoff: float) -> int:
    index = bisect_left(times, cutoff) - 1
    return positions[index] if index >= 0 else 0


def load_candles(path: Path) -> list[dict]:
    candles = []
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        if raw["symbol"] != "XBTUSD":
            raise ValueError("wrong market symbol")
        candle = {"end": timestamp(raw["timestamp"]), **{k: float(raw[k]) for k in ("open", "high", "low", "close", "volume")}}
        if not all(math.isfinite(v) for v in candle.values()):
            raise ValueError("nonfinite candle")
        if candle["low"] <= 0 or not candle["low"] <= candle["close"] <= candle["high"] or candle["open"] <= 0 or candle["volume"] < 0:
            raise ValueError("invalid candle")
        # BitMEX open = previous close and may lie outside this hour's high/low.
        candles.append(candle)
    candles.sort(key=lambda row: row["end"])
    if any(b["end"] - a["end"] != HOUR for a, b in zip(candles, candles[1:])):
        raise ValueError("duplicate or missing hourly candles")
    return candles


def split_for(cutoff: float) -> str | None:
    validation_start = timestamp("2021-01-01T00:00:00Z")
    test_start = timestamp("2021-07-01T00:00:00Z")
    end = timestamp("2022-01-01T00:00:00Z")
    if cutoff + HOUR <= validation_start:
        return "train"
    if cutoff - LOOKBACK * HOUR >= validation_start and cutoff + HOUR <= test_start:
        return "validation"
    if cutoff - LOOKBACK * HOUR >= test_start and cutoff + HOUR <= end:
        return "test"
    return None


def make_example(history: list[dict], previous: str, target: str, split: str, rng: random.Random) -> dict:
    if len(history) != LOOKBACK:
        raise ValueError("requires 96 completed candles")
    cutoff = history[-1]["end"]
    closes = [r["close"] for r in history]
    returns = [b / a - 1 for a, b in zip(closes, closes[1:])]
    deltas = [b-a for a, b in zip(closes[-15:], closes[-14:])]
    gain, loss = sum(max(x, 0) for x in deltas) / 14, sum(max(-x, 0) for x in deltas) / 14
    mean_volume = sum(r["volume"] for r in history[-21:-1]) / 20
    features = {"lastClose": closes[-1], "changePct": 100 * (closes[-1] / closes[0] - 1),
                "rsi14": 100-100/(1+gain/loss) if loss else (100 if gain else 50),
                "volatilityPct": 100*pstdev(returns),
                "volumeRatio": history[-1]["volume"]/mean_volume if mean_volume else None}
    features = {k: round(v, 6) if v is not None else None for k, v in features.items()}
    options = list(ACTIONS)
    rng.shuffle(options)
    job = {"state": {"symbol": "XBTUSD", "market": "BitMEX XBTUSD inverse perpetual, USD 1 face value per contract",
                     "interval_minutes": 60, "data_cutoff": iso(cutoff), "position_side_before_cutoff": previous,
                     "features": features, "volume_unit": "contracts",
                     "recent_closed_candles": [{"time": int(r["end"]-HOUR), **{k: r[k] for k in ("open", "high", "low", "close", "volume")}} for r in history[-24:]],
                     "missing": ["equity", "leverage", "order_book", "news", "future_executions"]},
           "instructions": INSTRUCTIONS,
           "options": [{"name": name, "description": {"long": "Positive signed contract position after the next hour.", "short": "Negative signed contract position after the next hour.", "flat": "Zero contract position after the next hour."}[name]} for name in options]}
    return {"job": job, "target_index": options.index(target), "target_action": target,
            "previous_action": previous, "cutoff": iso(cutoff), "split": split}


def prepare(source: Path, candles_path: Path, output: Path, train_count=4096, eval_count=128):
    paths = sorted(source.glob("aoa-execution-*.csv"))
    if len(paths) != 4:
        raise ValueError("expected the four supplied execution CSV files")
    if output.exists():
        raise ValueError("output directory already exists; use a new versioned directory")
    times, positions, reconciliation = read_positions(paths)
    candles = load_candles(candles_path)
    groups: dict[str, list[tuple]] = {name: [] for name in ("train", "validation", "test")}
    for index in range(LOOKBACK-1, len(candles)):
        cutoff = candles[index]["end"]
        if cutoff < times[0] or cutoff+HOUR > times[-1]:
            continue  # Do not treat unobserved coverage edges as deliberate inactivity.
        split = split_for(cutoff)
        if split:
            previous = side(position_before(times, positions, cutoff))
            target = side(position_before(times, positions, cutoff+HOUR))
            groups[split].append((index, previous, target))
    manifest = {"task": "NEXT_HOUR_POSITION_SIDE", "provenance": "user_supplied_aoa_execution_export",
                "model": MODEL, "revision": REVISION,
                "source_files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths],
                "market_file": {"sha256": sha256(candles_path), "source": "https://www.bitmex.com/api/v1/trade/bucketed", "candles": len(candles)},
                "reconciliation": reconciliation, "splits": {}, "files": {},
                "assumptions": ["Timezone-naive CSV execution timestamps interpreted as UTC; original timezone metadata unavailable.",
                                "Zero initial position, independently checked against every BTC funding quantity.",
                                "Position labels are before next-hour boundary; fills exactly at the boundary belong to the next window.",
                                "At least 96h purge between chronological splits; no future fills/prices in model input.",
                                "Pilot samples are deterministic uniform samples within each chronological split.",
                                "Source attribution is user-supplied; trader identity/authenticity not independently verified.",
                                "Position-side imitation is not the production spot-direction task, a profitability test or deployment approval."]}
    output.mkdir(parents=True)
    rng = random.Random(3407)
    for split, rows in groups.items():
        limit = train_count if split == "train" else eval_count
        if len(rows) < limit:
            raise ValueError(f"too few eligible {split} rows")
        selected = sorted(rng.sample(rows, limit))
        path = output / f"{split}.jsonl"
        with path.open("w") as handle:
            for index, previous, target in selected:
                row = make_example(candles[index-LOOKBACK+1:index+1], previous, target, split, rng)
                handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
        manifest["splits"][split] = {"count": len(selected), "eligible_count": len(rows),
            "eligible_labels": dict(Counter(x[2] for x in rows)), "sampled_labels": dict(Counter(x[2] for x in selected)),
            "sampled_transitions": sum(x[1] != x[2] for x in selected),
            "first_cutoff": iso(candles[selected[0][0]]["end"]), "last_cutoff": iso(candles[selected[-1][0]]["end"]),
            "eligible_persistence_accuracy": sum(x[1] == x[2] for x in rows)/len(rows)}
        manifest["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest["dataset_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.candles, args.output), indent=2))
