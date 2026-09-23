import pytest

from jev_inference import action_task
from jev_inference.action_api import decide
from jev_inference.settings import Settings, parse_hold_margin

FAMILY = {"flat": -0.68, "position": -0.25}


def test_margin_for_selects_state_family():
    assert action_task.margin_for(FAMILY, "flat") == -0.68
    assert action_task.margin_for(FAMILY, "long") == -0.25
    assert action_task.margin_for(FAMILY, "short") == -0.25
    assert action_task.margin_for(0.1, "flat") == 0.1


@pytest.mark.parametrize("bad", [{"flat": 1.0}, {"flat": 1.0, "position": float("nan")}, {"flat": 1, "position": 99}, True])
def test_check_margin_rejects_malformed(bad):
    with pytest.raises(ValueError):
        action_task.check_margin(bad)


def test_env_accepts_json_family_margin_and_settings_validate():
    parsed = parse_hold_margin('{"flat": -0.68, "position": -0.25}')
    assert parsed == FAMILY
    Settings(api_key="x" * 40, action_hold_margin=parsed)
    with pytest.raises(ValueError):
        Settings(api_key="x" * 40, action_hold_margin={"flat": -0.68})


def test_negative_margin_makes_hold_win_more_often():
    names, logits = ["hold", "open_long", "open_short"], [0.0, 0.5, -1.0]
    assert decide(names, logits, 0.0) == "open_long"
    assert decide(names, logits, action_task.margin_for(FAMILY, "flat")) == "hold"
