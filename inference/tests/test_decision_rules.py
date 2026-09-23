import random

import pytest

from jev_inference import action_task, decision_rules
from jev_inference.action_api import decide

STEP = action_task.STEP
FEAT = {"return_1h_pct": 0.1, "return_4h_pct": 0.5, "return_24h_pct": -1.0}
FLAT = action_task.flat_position()


def pos(side, opened=None, last=None):
    return dict(side=side, units=1.0 if side != "flat" else 0.0, entry_price=100.0 if side != "flat" else None,
                opened_at=opened, last_trade_at=last)


def test_empty_rules_equal_plain_margin_decision():
    rng = random.Random(1)
    for _ in range(500):
        side = rng.choice(["flat", "long", "short"])
        names = list(action_task.options_for(side))
        logits = [rng.uniform(-3, 3) for _ in names]
        m = rng.choice([0.0, -0.5, {"flat": -0.68, "position": -0.25}])
        got = decision_rules.decide(names, logits, rules={"margins": m}, position=pos(side, 0, 0), cutoff=10 * STEP, features=FEAT)
        assert got == decide(names, logits, action_task.margin_for(m, side))


def test_cooldown_min_hold_trend_allowed_and_prob_floor():
    flat_names, flat_logits = ["hold", "open_long", "open_short"], [0.0, 2.0, 1.0]
    c = 100 * STEP
    assert decision_rules.decide(flat_names, flat_logits, rules={}, position=FLAT, cutoff=c, features=FEAT) == "open_long"
    cooled = pos("flat", None, c - 3 * STEP)
    assert decision_rules.decide(flat_names, flat_logits, rules={"cooldown_bars": 4}, position=cooled, cutoff=c, features=FEAT) == "hold"
    # 24h trend negative: longs blocked, short allowed
    assert decision_rules.decide(flat_names, flat_logits, rules={"trend_filter": "return_24h_pct"}, position=FLAT, cutoff=c, features=FEAT) == "open_short"
    assert decision_rules.decide(flat_names, flat_logits, rules={"open_prob_min": 0.9}, position=FLAT, cutoff=c, features=FEAT) == "hold"
    names, logits = ["hold", "add", "reduce", "close"], [0.0, 0.1, 0.5, 2.0]
    young = pos("long", c - 2 * STEP, c - 2 * STEP)
    assert decision_rules.decide(names, logits, rules={"min_hold_bars": 4}, position=young, cutoff=c, features=FEAT) == "add"
    assert decision_rules.decide(names, logits, rules={"allowed": ["open_long", "open_short", "close"], "min_hold_bars": 4},
                                 position=young, cutoff=c, features=FEAT) == "hold"


@pytest.mark.parametrize("bad", [{"nope": 1}, {"allowed": ["hold"]}, {"min_hold_bars": -1}, {"open_prob_min": 2}, {"trend_filter": "x"}])
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        decision_rules.validate(bad)
