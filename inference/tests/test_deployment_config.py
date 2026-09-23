import json
from pathlib import Path

import pytest

from jev_inference import deployment

RECORD = json.loads((Path(__file__).resolve().parents[1] / "ADAPTER_DEPLOYMENT.json").read_text())


def test_deployed_adapter_pin_is_unchanged():
    entry = deployment.SERVING[RECORD["modalApp"]]
    assert (entry["adapter_id"], entry["adapter_revision"], entry["adapter_sha256"]) == (
        RECORD["adapter"], RECORD["adapterRevision"], RECORD["adapterSha256"])
    assert entry["action_hold_margin"] == 0.0
    assert deployment.image_env("jev-nyotti") == {
        "ACTION_HOLD_MARGIN": "0.0",
        "LORA_MODEL_ID": RECORD["adapter"],
        "LORA_REVISION": RECORD["adapterRevision"],
        "LORA_SHA256": RECORD["adapterSha256"],
    }


def test_base_app_loads_no_adapter():
    assert deployment.image_env("jev-qwen-35-4b") == {"ACTION_HOLD_MARGIN": "0.0"}


def test_invalid_entries_fail_before_deploy(monkeypatch):
    bad = dict(deployment.SERVING)
    bad["x"] = dict(bad["jev-nyotti"], adapter_revision="main")
    bad["y"] = dict(bad["jev-nyotti"], action_hold_margin=float("nan"))
    monkeypatch.setattr(deployment, "SERVING", bad)
    for name in ("x", "y"):
        with pytest.raises(ValueError):
            deployment.image_env(name)


def test_numeric_cpu_app_refuses_until_pinned(monkeypatch):
    with pytest.raises(ValueError, match="pin numeric_model_sha256"):
        deployment.numeric_image_env("jev-nyotti-action-cpu")
    monkeypatch.setitem(deployment.NUMERIC_SERVING, "jev-nyotti-action-cpu",
                        dict(numeric_model_sha256="a" * 64, action_hold_margin=0.75))
    assert deployment.numeric_image_env("jev-nyotti-action-cpu") == {
        "ACTION_POLICY": "numeric", "INFERENCE_DEVICE": "cpu",
        "NUMERIC_MODEL_PATH": deployment.NUMERIC_MODEL_REMOTE_PATH,
        "NUMERIC_MODEL_SHA256": "a" * 64, "ACTION_HOLD_MARGIN": "0.75",
    }
    monkeypatch.setitem(deployment.NUMERIC_SERVING, "jev-nyotti-action-cpu",
                        dict(numeric_model_sha256="main", action_hold_margin=0.75))
    with pytest.raises(ValueError):
        deployment.numeric_image_env("jev-nyotti-action-cpu")
