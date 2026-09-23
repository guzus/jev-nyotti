"""Stateful ACTION_V1 historical replay core. Pure: no network, GPU or Modal imports.

Each symbol starts flat and carries the model's OWN paper position through
`action_task.apply_action`. At every 15-minute cutoff the job sees only the 96
candles that closed at or before the cutoff; the fill price is the close of the
candle ending at the cutoff. The forward candle opening at the cutoff is published
as the execution/marking candle and never reaches the model.

The Modal wrapper (`modal_action_replay.py`) owns budget, watchdog and volume I/O;
this module owns validation, the decision loop and checkpoint/resume integrity.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from pathlib import Path

from . import action_task
from .action_api import check_candle_open, decide, softmax
from .replay import epoch, fingerprint, iso

STEP = action_task.STEP
LOOKBACK = action_task.LOOKBACK
SYMBOL = re.compile(r"[A-Za-z0-9._/-]{1,40}")
CONTRACT_SHA256 = hashlib.sha256(Path(action_task.__file__).read_bytes()).hexdigest()


def budget_seconds(budget_usd: float, rate_usd_second: float, reserve_seconds: float, cap: float = 6900) -> float:
    if not math.isfinite(budget_usd) or not 0.32 <= budget_usd <= 10:
        raise ValueError("budget must be 0.32..10 USD")
    seconds = min(cap, budget_usd / rate_usd_second - reserve_seconds)
    if seconds < 60:
        raise ValueError("budget must reserve at least 60 inference seconds")
    return seconds


def index_candles(candles: list[dict]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for c in candles:
        if set(c) != {"time", "open", "high", "low", "close", "volume"}:
            raise ValueError("candle requires exactly time, open, high, low, close, volume")
        t = c["time"]
        if isinstance(t, bool) or not isinstance(t, int) or t % STEP or t in result:
            raise ValueError("duplicate or unaligned 15m candle")
        values = [c[k] for k in ("open", "high", "low", "close", "volume")]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError("nonfinite candle")
        if c["low"] <= 0 or c["high"] < c["low"] or not c["low"] <= c["close"] <= c["high"] or c["volume"] < 0:
            raise ValueError("invalid OHLCV")
        check_candle_open([c])
        result[t] = c
    return result


def lookback(index: dict[int, dict], cutoff: int) -> list[dict]:
    """The 96 candles whose close is <= cutoff (open times cutoff-96*STEP .. cutoff-STEP)."""
    try:
        return [index[t] for t in range(cutoff - LOOKBACK * STEP, cutoff, STEP)]
    except KeyError as e:
        raise ValueError(f"missing required 15m candle {iso(e.args[0])}") from None


def forward(index: dict[int, dict], cutoff: int) -> dict:
    """Execution/marking candle [cutoff, cutoff+15m). Output only; never model input."""
    try:
        return index[cutoff]
    except KeyError:
        raise ValueError(f"missing execution 15m candle {iso(cutoff)}") from None


class Plan:
    """Validated immutable replay input. Every cutoff's coverage is checked up front."""

    def __init__(self, manifest: dict, *, now: float):
        if not isinstance(manifest.get("source"), str) or not manifest["source"].strip() or len(manifest["source"]) > 200:
            raise ValueError("source is required (<= 200 chars)")
        extra = set(manifest) - {"source", "from", "to", "series", "provenance"}
        if extra:
            raise ValueError("unexpected manifest keys")
        self.source = manifest["source"]
        self.start, self.end = epoch(manifest["from"]), epoch(manifest["to"])
        if self.start % STEP or self.end % STEP or self.end <= self.start or self.end > now:
            raise ValueError("range must be completed, ordered UTC 15-minute boundaries")
        series = manifest.get("series")
        if not isinstance(series, list) or not series:
            raise ValueError("nonempty series is required")
        self.symbols: list[str] = []
        self.markets: dict[str, str] = {}
        self.indexes: dict[str, dict[int, dict]] = {}
        for s in series:
            if set(s) - {"symbol", "candles", "market"}:
                raise ValueError("unexpected series keys")
            symbol = s.get("symbol")
            if not isinstance(symbol, str) or not SYMBOL.fullmatch(symbol) or symbol in self.indexes:
                raise ValueError("invalid or duplicate symbol")
            market = s.get("market") or action_task.market_label(re.sub(r"USDT?$", "", symbol))
            if not isinstance(market, str) or not market.strip() or len(market) > 120:
                raise ValueError("market label must be 1..120 chars")
            index = index_candles(s["candles"])
            for cutoff in self.cutoffs():
                lookback(index, cutoff)
                forward(index, cutoff)
            self.symbols.append(symbol)
            self.markets[symbol] = market
            self.indexes[symbol] = index

    def cutoffs(self, first: int = 0) -> range:
        return range(self.start + first * STEP, self.end, STEP)

    def job(self, symbol: str, cutoff: int, position: dict) -> dict:
        return action_task.build_job(candles=lookback(self.indexes[symbol], cutoff), cutoff=cutoff,
                                     position=position, market=self.markets[symbol])

    def price(self, symbol: str, cutoff: int) -> float:
        return lookback(self.indexes[symbol], cutoff)[-1]["close"]


def identity(manifest: dict, *, adapter_id: str, adapter_revision: str, adapter_sha256: str,
             base_revision: str, hold_margin: float, policy: str = "lora", numeric_sha256: str | None = None) -> str:
    fields = dict(input=manifest, task=action_task.TASK, contractSha256=CONTRACT_SHA256,
                  adapter=dict(id=adapter_id, revision=adapter_revision, sha256=adapter_sha256),
                  baseRevision=base_revision, holdMargin=hold_margin)
    if policy != "lora":  # lora identities stay unchanged so existing checkpoints still resume
        fields.update(policy=policy, numericSha256=numeric_sha256)
    return fingerprint(fields)


def numeric_scorer(model: dict) -> "Scorer":
    """ACTION_POLICY=numeric replay scorer: the same numeric_policy.log_probs used by live /action."""
    from . import numeric_policy

    def score(jobs: list[dict]) -> list:
        out = []
        for job in jobs:
            logps = numeric_policy.log_probs(model, job["state"])
            out.append(NumericScore(logits=[logps[o["name"]] for o in job["options"]], inputTokens=0))
        return out
    return score


class NumericScore:
    def __init__(self, logits: list[float], inputTokens: int):
        self.logits, self.inputTokens = logits, inputTokens


def new_state(run_identity: str) -> dict:
    return dict(identity=run_identity, records=[], completedCutoffs=0)


def _choose(names: list[str], logits: list[float], spec, position: dict, cutoff: int, features) -> str:
    """spec: a hold margin (number / per-family dict) or {"rules": decision_rules} (ACTION_V5+)."""
    if isinstance(spec, dict) and "rules" in spec:
        from . import decision_rules
        return decision_rules.decide(names, logits, rules=spec["rules"], position=position, cutoff=cutoff,
                                     features=features() if callable(features) else features)
    return decide(names, logits, action_task.margin_for(spec, position["side"]))


def restore_positions(plan: Plan, state: dict, margin: float) -> dict[str, dict]:
    """Rebuild every symbol's paper position from checkpoint records, verifying the chain."""
    positions = {symbol: action_task.flat_position() for symbol in plan.symbols}
    records = state["records"]
    if len(records) != state["completedCutoffs"] * len(plan.symbols):
        raise ValueError("checkpoint holds a partial cutoff")
    for i, record in enumerate(records):
        cutoff = plan.start + (i // len(plan.symbols)) * STEP
        symbol = plan.symbols[i % len(plan.symbols)]
        if record["symbol"] != symbol or record["marketAsOf"] != iso(cutoff) or cutoff >= plan.end:
            raise ValueError("checkpoint order corrupted")
        position = positions[symbol]
        if record["options"] != list(action_task.options_for(position["side"])):
            raise ValueError("checkpoint options do not match carried position")
        if _choose(record["options"], record["logits"], margin, position, cutoff,
                   lambda: plan.job(symbol, cutoff, position)["state"]["features"]) != record["action"]:
            raise ValueError("checkpoint decision does not match logits and margin")
        if record["price"] != plan.price(symbol, cutoff):
            raise ValueError("checkpoint price does not match input")
        positions[symbol] = action_task.apply_action(position, record["action"], record["price"], cutoff)
        if (positions[symbol]["side"], positions[symbol]["units"]) != (record["sideAfter"], record["unitsAfter"]):
            raise ValueError("checkpoint position chain corrupted")
    return positions


Scorer = Callable[[list[dict]], list]  # job dicts -> objects with .logits and .inputTokens


def step(plan: Plan, positions: dict[str, dict], cutoff: int, score: Scorer, margin: float) -> list[dict]:
    """Decide one cutoff for every symbol. Each job depends only on its own symbol's state."""
    jobs = [plan.job(symbol, cutoff, positions[symbol]) for symbol in plan.symbols]
    scores = score(jobs)
    if len(scores) != len(jobs):
        raise RuntimeError("scorer returned mismatched results")
    records = []
    for symbol, job, result in zip(plan.symbols, jobs, scores, strict=True):
        names = [o["name"] for o in job["options"]]
        logits = [float(v) for v in result.logits]
        action = _choose(names, logits, margin, positions[symbol], cutoff, job["state"]["features"])
        price = plan.price(symbol, cutoff)
        after = action_task.apply_action(positions[symbol], action, price, cutoff)
        records.append(dict(symbol=symbol, marketAsOf=iso(cutoff), sideBefore=positions[symbol]["side"], options=names,
                            logits=logits, action=action, price=price, sideAfter=after["side"], unitsAfter=after["units"],
                            execution=forward(plan.indexes[symbol], cutoff), inputTokens=result.inputTokens))
    return records


def run(plan: Plan, state: dict, score: Scorer, margin: float, *, max_decisions: int,
        may_continue: Callable[[], bool] = lambda: True, on_cutoff: Callable[[], None] = lambda: None) -> int:
    """Advance whole cutoffs until done, the decision cap, or `may_continue()` is False.

    Only complete cutoffs are appended to `state`; `on_cutoff` persists after each one.
    Returns the number of new decisions.
    """
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    positions = restore_positions(plan, state, margin)
    calls = 0
    for cutoff in plan.cutoffs(state["completedCutoffs"]):
        if calls + len(plan.symbols) > max_decisions or not may_continue():
            break
        records = step(plan, positions, cutoff, score, margin)
        state["records"].extend(records)
        state["completedCutoffs"] += 1
        for record in records:
            positions[record["symbol"]] = action_task.apply_action(positions[record["symbol"]], record["action"], record["price"], cutoff)
        calls += len(records)
        on_cutoff()
    return calls


def output(plan: Plan, state: dict, *, model: str, revision: str, margin: float, generated_at: float,
           policy: str = "lora") -> dict:
    completed_end = plan.start + state["completedCutoffs"] * STEP
    result = dict(model=model, policy=policy, revision=revision, task=action_task.TASK, intervalMinutes=action_task.INTERVAL_MINUTES,
                  holdMargin=margin, **{"from": iso(plan.start), "to": iso(completed_end)}, source=plan.source,
                  generatedAt=iso(generated_at), series=[])
    for symbol in plan.symbols:
        rows = [r for r in state["records"] if r["symbol"] == symbol]
        result["series"].append(dict(
            symbol=symbol,
            decisions=[dict(marketAsOf=r["marketAsOf"], action=r["action"], sideAfter=r["sideAfter"],
                            unitsAfter=r["unitsAfter"], price=r["price"],
                            probabilities=dict(zip(r["options"], softmax(r["logits"]))))
                       for r in rows],
            candles=[r["execution"] for r in rows],
        ))
    return result
