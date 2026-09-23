import copy
import json
from types import SimpleNamespace

import pytest

from jev_inference import action_replay, action_task
from jev_inference.replay import iso

STEP = action_task.STEP
START = 1_700_000_100 - 1_700_000_100 % STEP + 200 * STEP


def candles(first: int, count: int) -> list[dict]:
    rows = []
    for i in range(count):
        base = 100 + (i % 7) - (i % 3) * 0.5
        rows.append(dict(time=first + i * STEP, open=base, high=base + 2, low=base - 2, close=base + 0.5, volume=10 + i % 5))
    return rows


def manifest(steps: int = 6, symbols=("BTCUSDT", "ETHUSDT")) -> dict:
    series = [dict(symbol=s, candles=candles(START - 96 * STEP, 96 + steps)) for s in symbols]
    return dict(source="binance-spot-15m", **{"from": iso(START), "to": iso(START + steps * STEP)}, series=series)


class Scripted:
    """Fake engine: open_long when flat, then add, reduce, close in turn; records every job."""

    def __init__(self):
        self.jobs = []

    def __call__(self, jobs):
        self.jobs.extend(jobs)
        out = []
        for job in jobs:
            names = [o["name"] for o in job["options"]]
            want = "open_long" if "open_long" in names else ["add", "reduce", "close"][len(self.jobs) % 3]
            out.append(SimpleNamespace(logits=[1.0 if n == want else 0.0 for n in names], inputTokens=123))
        return out


def plan(m=None):
    return action_replay.Plan(m or manifest(), now=START + 10_000 * STEP)


def test_every_prompt_is_action_task_build_job_on_closed_candles_only():
    p, fake = plan(), Scripted()
    state = action_replay.new_state("x")
    action_replay.run(p, state, fake, 0.0, max_decisions=100)
    assert state["completedCutoffs"] == 6 and len(state["records"]) == 12
    index = p.indexes["BTCUSDT"]
    first = fake.jobs[0]
    lookback = [index[t] for t in range(START - 96 * STEP, START, STEP)]
    assert first == action_task.build_job(candles=lookback, cutoff=START, position=action_task.flat_position(),
                                          market="BTCUSDT on binance-spot-15m")
    assert first["state"]["data_cutoff"] == iso(START)
    # Execution price is the close of the candle ending at the cutoff, never the forward candle.
    record = state["records"][0]
    assert record["price"] == index[START - STEP]["close"] and record["execution"] == index[START]


def test_forward_candle_cannot_change_the_decision_input():
    m = manifest()
    changed = copy.deepcopy(m)
    for c in changed["series"][0]["candles"]:
        if c["time"] >= START:
            c.update(open=500, high=900, low=400, close=800, volume=1e6)
    a, b = plan(m), plan(changed)
    assert a.job("BTCUSDT", START, action_task.flat_position()) == b.job("BTCUSDT", START, action_task.flat_position())


def test_model_carries_its_own_paper_position():
    p, fake = plan(manifest(symbols=("BTCUSDT",))), Scripted()
    state = action_replay.new_state("x")
    action_replay.run(p, state, fake, 0.0, max_decisions=100)
    actions = [r["action"] for r in state["records"]]
    assert actions[0] == "open_long"
    assert [r["sideBefore"] for r in state["records"]][1] == "long"
    assert fake.jobs[1]["state"]["position"]["side"] == "long"
    after = [(r["sideAfter"], r["unitsAfter"]) for r in state["records"]]
    position = action_task.flat_position()
    for r, cutoff in zip(state["records"], p.cutoffs()):
        position = action_task.apply_action(position, r["action"], r["price"], cutoff)
        assert (position["side"], position["units"]) == after.pop(0)


def test_hold_margin_only_penalizes_hold():
    def score(jobs):
        return [SimpleNamespace(logits=[2.0 if o["name"] == "hold" else 1.0 for o in j["options"]], inputTokens=1) for j in jobs]
    p = plan()
    for margin, expected in ((0.0, "hold"), (1.5, "open_long")):
        state = action_replay.new_state("x")
        action_replay.run(p, state, score, margin, max_decisions=2)
        assert state["records"][0]["action"] == expected


def test_resume_from_checkpoint_matches_uninterrupted_run():
    p = plan()
    whole = action_replay.new_state("x")
    action_replay.run(p, whole, Scripted(), 0.0, max_decisions=100)
    part = action_replay.new_state("x")
    fake = Scripted()
    assert action_replay.run(p, part, fake, 0.0, max_decisions=5) == 4  # whole cutoffs only
    part = json.loads(json.dumps(part))
    action_replay.run(p, part, fake, 0.0, max_decisions=100)
    assert part == json.loads(json.dumps(whole))


def test_tampered_checkpoint_is_rejected():
    p = plan()
    state = action_replay.new_state("x")
    action_replay.run(p, state, Scripted(), 0.0, max_decisions=4)
    for field, value in (("action", "hold"), ("price", 1.0), ("unitsAfter", 3.0), ("symbol", "ETHUSDT")):
        broken = copy.deepcopy(state)
        broken["records"][2][field] = value
        with pytest.raises(ValueError):
            action_replay.restore_positions(p, broken, 0.0)
    with pytest.raises(ValueError):
        action_replay.restore_positions(p, state, -5.0)  # margin changed under the same logits


def test_gaps_and_unfinished_ranges_fail_before_any_scoring():
    gap = manifest()
    del gap["series"][1]["candles"][40]
    with pytest.raises(ValueError, match="missing required"):
        plan(gap)
    no_forward = manifest()
    no_forward["series"][0]["candles"].pop()
    with pytest.raises(ValueError, match="missing execution"):
        plan(no_forward)
    with pytest.raises(ValueError, match="completed"):
        action_replay.Plan(manifest(), now=START + 5 * STEP)
    unaligned = manifest()
    unaligned["from"] = iso(START + 60)
    with pytest.raises(ValueError):
        plan(unaligned)
    bad = manifest()
    bad["series"][0]["candles"][3]["open"] = 1e9
    with pytest.raises(ValueError, match="OHLCV"):
        plan(bad)


def test_output_shape_and_identity_binds_model_and_margin():
    p = plan()
    state = action_replay.new_state("x")
    action_replay.run(p, state, Scripted(), 0.25, max_decisions=100)
    out = action_replay.output(p, state, model="Qwen/Qwen3.5-4B", revision="r@a", margin=0.25, generated_at=START + 9e5)
    assert {k: out[k] for k in ("task", "intervalMinutes", "holdMargin", "from", "to", "source")} == dict(
        task="ACTION_V1", intervalMinutes=15, holdMargin=0.25, **{"from": iso(START), "to": iso(START + 6 * STEP)},
        source="binance-spot-15m")
    first = out["series"][0]
    assert len(first["decisions"]) == len(first["candles"]) == 6
    d = first["decisions"][0]
    assert set(d) == {"marketAsOf", "action", "sideAfter", "unitsAfter", "price", "probabilities"}
    assert set(d["probabilities"]) == {"hold", "open_long", "open_short"} and abs(sum(d["probabilities"].values()) - 1) < 1e-9
    assert first["candles"][0]["time"] == START
    ids = dict(adapter_id="a/b", adapter_revision="1" * 40, adapter_sha256="2" * 64, base_revision="main")
    m = manifest()
    assert action_replay.identity(m, hold_margin=0.0, **ids) != action_replay.identity(m, hold_margin=0.5, **ids)
    assert action_replay.identity(m, hold_margin=0.0, **ids) != action_replay.identity(m, hold_margin=0.0, **{**ids, "adapter_revision": "3" * 40})


def test_budget_bounds():
    assert action_replay.budget_seconds(1.0, 0.0013, 180) == pytest.approx(1 / 0.0013 - 180)
    for bad in (0.1, 11, float("nan")):
        with pytest.raises(ValueError):
            action_replay.budget_seconds(bad, 0.0013, 180)
