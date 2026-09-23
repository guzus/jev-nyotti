import copy
import json
import math

import pytest
from fastapi.testclient import TestClient

from jev_inference import action_task
from jev_inference.action_api import ActionRequest, decide, job_from_request, softmax, to_scoring_job
from jev_inference.app import create_app
from jev_inference.engine import InputTooLong
from jev_inference.labels import Label
from jev_inference.prompt import messages_for_job
from jev_inference.schemas import JobScore
from jev_inference.settings import MAX_BODY_BYTES, MODEL_ID, MODEL_REVISION, Settings, parse_hold_margin

KEY = "test-only-key-012345678901234567890123456789"
HEADERS = {"Authorization": f"Bearer {KEY}"}
CUTOFF = 1_758_585_600  # 2025-09-23T00:00:00Z, a 15-minute boundary in the past
CUTOFF_ISO = "2025-09-23T00:00:00Z"


def candles(cutoff=CUTOFF, base=112_345.67):
    rows = []
    for i in range(action_task.LOOKBACK):
        t = cutoff - (action_task.LOOKBACK - i) * action_task.STEP
        o = base + 13.37 * math.sin(i / 3)
        c = o * (1 + 0.0007 * math.cos(i / 5))
        rows.append(dict(time=t, open=round(o, 2), high=round(max(o, c) * 1.0009, 2),
                         low=round(min(o, c) * 0.9991, 2), close=round(c, 2), volume=round(12.345678 + i % 7, 6)))
    return rows


def body(**overrides):
    value = dict(market="Kraken XBT/USD (untested transfer)", cutoff=CUTOFF_ISO, candles=candles(),
                 position=dict(side="flat", entry_price=None, opened_at=None, last_trade_at=None))
    value.update(overrides)
    return value


LONG = dict(side="long", entry_price=111_000.5, opened_at=CUTOFF - 7200, last_trade_at=CUTOFF - 3600)


class FakeEngine:
    def __init__(self, logits=None):
        self.loaded = False
        self.calls = 0
        self.jobs = []
        self.logits = logits

    def load(self):
        self.loaded = True

    def prepare(self, jobs):
        assert self.loaded
        self.jobs.extend(jobs)
        if jobs[0].state.get("market") == "too_long":
            raise InputTooLong()
        if jobs[0].state.get("market") == "error":
            raise RuntimeError(f"do not disclose {KEY}")
        return jobs

    def score(self, jobs):
        self.calls += 1
        return [JobScore(logits=list(self.logits or [float(i) for i in range(len(job.options))]), inputTokens=321) for job in jobs]


def client_for(engine, **settings):
    return TestClient(create_app(Settings(api_key=KEY, **settings), engine_factory=lambda: engine))


@pytest.fixture
def running():
    engine = FakeEngine()
    with client_for(engine) as client:
        yield client, engine


def test_flat_action_builds_contract_job_and_reports_raw_softmax(running):
    client, engine = running
    response = client.post("/action", headers=HEADERS, json=body())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model"] == MODEL_ID and result["revision"] == MODEL_REVISION and result["task"] == "ACTION_V1"
    assert [o["name"] for o in result["options"]] == list(action_task.FLAT_OPTIONS)
    assert result["action"] == "open_short"  # fake logits [0, 1, 2]
    expected = softmax([0.0, 1.0, 2.0])
    assert [o["probability"] for o in result["options"]] == pytest.approx(expected)
    assert result["holdMargin"] == 0.0 and result["inputTokens"] == 321 and result["elapsedMs"] >= 0
    # The scored job is exactly action_task.build_job output for the same inputs.
    expected_job = action_task.build_job(candles=candles(), cutoff=CUTOFF, market=body()["market"],
                                         position=dict(side="flat", entry_price=None, opened_at=None, last_trade_at=None))
    assert engine.jobs[0].model_dump() == to_scoring_job(expected_job).model_dump()
    assert engine.calls == 1


def test_position_options_and_margin_subtracts_from_hold_only():
    engine = FakeEngine(logits=[2.0, 1.5, 0.0, -1.0])
    with client_for(engine, action_hold_margin=0.6) as client:
        result = client.post("/action", headers=HEADERS, json=body(position=LONG)).json()
    assert [o["name"] for o in result["options"]] == list(action_task.POSITION_OPTIONS)
    assert result["action"] == "add" and result["holdMargin"] == 0.6
    # Probabilities stay the softmax of the raw (unshifted) logits.
    assert [o["probability"] for o in result["options"]] == pytest.approx(softmax([2.0, 1.5, 0.0, -1.0]))
    state = engine.jobs[0].state["position"]
    assert state["side"] == "long" and state["position_age_minutes"] == 120.0
    assert state["minutes_since_last_execution"] == 60.0
    engine = FakeEngine(logits=[2.0, 1.5, 0.0, -1.0])
    with client_for(engine, action_hold_margin=0.5) as client:
        assert client.post("/action", headers=HEADERS, json=body(position=LONG)).json()["action"] == "hold"  # tie -> first max


def test_decide_ties_negative_margin_and_hold_lookup_by_name():
    assert decide(["open_long", "hold"], [1.0, 1.0], 0.0) == "open_long"
    assert decide(["open_long", "hold"], [1.0, 0.9], -0.2) == "hold"
    with pytest.raises(ValueError):
        decide(["hold"], [float("nan")], 0.0)


def test_prompt_round_trip_is_exactly_build_job_state():
    """Guard against train/serve drift: what the model reads equals build_job output byte-for-byte."""
    job = action_task.build_job(candles=candles(), cutoff=CUTOFF, market="m", position=LONG)
    labels = [Label(text=t, token_id=i) for i, t in enumerate("ABCD")]
    content = json.loads(messages_for_job(to_scoring_job(job), labels)[1]["content"])
    dumped = json.dumps(content["state"], ensure_ascii=False, separators=(",", ":"))
    assert dumped == json.dumps(job["state"], ensure_ascii=False, separators=(",", ":"))
    assert content["instructions"] == action_task.INSTRUCTIONS
    assert [o["name"] for o in content["options"]] == list(action_task.POSITION_OPTIONS)
    assert "112345" not in dumped  # no absolute price reaches the model


def test_integer_prices_are_accepted_and_identical_to_floats(running):
    client, engine = running
    rows = candles()
    ints = [dict(r, open=round(r["open"]), high=math.ceil(r["high"]), low=math.floor(r["low"]), close=round(r["close"]), volume=7) for r in rows]
    for r in ints:
        r["close"] = min(max(r["close"], r["low"]), r["high"])
        r["open"] = min(max(r["open"], r["low"]), r["high"])
    assert client.post("/action", headers=HEADERS, json=body(candles=ints)).status_code == 200
    floats = [dict(r, **{k: float(r[k]) for k in ("open", "high", "low", "close", "volume")}) for r in ints]
    assert client.post("/action", headers=HEADERS, json=body(candles=floats)).status_code == 200
    assert engine.jobs[0].model_dump() == engine.jobs[1].model_dump()


def test_realistic_96_candle_body_is_well_under_64kib():
    rows = [dict(r, open=r["open"] + 0.123456789012, close=r["close"] + 0.123456789012, volume=123456.789012345678) for r in candles()]
    payload = json.dumps(body(candles=rows, position=LONG, market="x" * 120)).encode()
    assert len(payload) < MAX_BODY_BYTES / 3, len(payload)


def mutation(fn):
    def apply(value):
        fn(value)
        return value
    return apply


@pytest.mark.parametrize("mutate", [
    mutation(lambda b: b["candles"].pop()),                                   # 95 candles
    mutation(lambda b: b["candles"].append(dict(b["candles"][-1], time=CUTOFF))),  # 97 candles
    mutation(lambda b: b.update(candles=candles(CUTOFF + action_task.STEP)[:96])),  # ends after cutoff
    mutation(lambda b: b["candles"][10].update(time=b["candles"][10]["time"] + 60)),  # gap / misaligned
    mutation(lambda b: b["candles"][5].update(open=b["candles"][5]["high"] * 2)),    # open outside range
    mutation(lambda b: b["candles"][5].update(close=0.0, low=0.0)),
    mutation(lambda b: b["candles"][5].update(volume=-1)),
    mutation(lambda b: b["candles"][5].update(volume=True)),
    mutation(lambda b: b["candles"][5].update(open="1")),
    mutation(lambda b: b["candles"][5].update(time=float(b["candles"][5]["time"]))),
    mutation(lambda b: b["candles"][5].update(extra=1)),
    mutation(lambda b: b.update(cutoff="2025-09-23T00:05:00Z")),
    mutation(lambda b: b.update(cutoff="2025-09-23T00:00:00+00:00")),
    mutation(lambda b: b.update(cutoff="2025-09-23 00:00:00Z")),
    mutation(lambda b: b.update(cutoff=CUTOFF)),
    mutation(lambda b: b.update(market="")),
    mutation(lambda b: b.update(market="   ")),
    mutation(lambda b: b.update(market="x" * 121)),
    mutation(lambda b: b.update(extra="field")),
    mutation(lambda b: b["position"].update(side="LONG")),
    mutation(lambda b: b["position"].pop("last_trade_at")),
    mutation(lambda b: b["position"].update(entry_price=100.0)),             # flat with entry
    mutation(lambda b: b["position"].update(last_trade_at=CUTOFF + 1)),      # state from the future
    mutation(lambda b: b.update(position=dict(LONG, entry_price=None))),
    mutation(lambda b: b.update(position=dict(LONG, entry_price=0))),
    mutation(lambda b: b.update(position=dict(LONG, opened_at=None))),
    mutation(lambda b: b.update(position=dict(LONG, opened_at=CUTOFF + 900))),
    mutation(lambda b: b.update(position=dict(LONG, last_trade_at=LONG["opened_at"] - 1))),
    mutation(lambda b: b.update(position=dict(LONG, units=2))),
])
def test_invalid_action_requests_are_422_without_gpu_work(running, mutate):
    client, engine = running
    response = client.post("/action", headers=HEADERS, json=mutate(copy.deepcopy(body())))
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_request"
    assert engine.calls == 0 and engine.jobs == []


@pytest.mark.parametrize("content", [b"{", b"\xff", b"[]", b'{"market":NaN}',
                                     json.dumps(body()).replace('"volume": 12.345678', '"volume": 1e400', 1).encode()])
def test_malformed_or_nonfinite_action_json_is_rejected(running, content):
    client, engine = running
    assert client.post("/action", headers=HEADERS, content=content).status_code == 422
    assert engine.calls == 0


def test_future_cutoff_is_rejected():
    request = ActionRequest.model_validate(body())
    with pytest.raises(ValueError, match="future"):
        job_from_request(request, now=CUTOFF - 3600)
    assert job_from_request(request, now=CUTOFF - 30)[1] == CUTOFF  # small clock skew tolerated


def test_action_requires_auth_and_readiness():
    engine = FakeEngine()
    app = create_app(Settings(api_key=KEY), engine_factory=lambda: engine)
    assert TestClient(app).post("/action", headers=HEADERS, json=body()).status_code == 503
    with TestClient(app) as client:
        assert client.post("/action", json=body()).status_code == 401
        assert client.post("/action", headers={"Authorization": "Bearer wrong"}, json=body()).status_code == 401
    assert engine.calls == 0


def test_engine_failures_are_sanitized(running):
    client, engine = running
    assert client.post("/action", headers=HEADERS, json=body(market="too_long")).json()["error"]["code"] == "input_too_long"
    response = client.post("/action", headers=HEADERS, json=body(market="error"))
    assert response.status_code == 503 and KEY not in response.text


def test_mismatched_engine_scores_fail_closed():
    engine = FakeEngine(logits=[1.0, 2.0])
    with client_for(engine) as client:
        assert client.post("/action", headers=HEADERS, json=body()).status_code == 503


def test_health_reports_action_margin():
    with client_for(FakeEngine(), action_hold_margin=-1.25) as client:
        assert client.get("/healthz").json()["action"] == {"task": "ACTION_V1", "holdMargin": -1.25}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 20.0001, -21, True, "1"])
def test_invalid_hold_margin_rejected(value):
    with pytest.raises(ValueError, match="ACTION_HOLD_MARGIN"):
        Settings(api_key=KEY, action_hold_margin=value)


def test_hold_margin_env_parsing(monkeypatch):
    assert parse_hold_margin(None) == 0.0 and parse_hold_margin(" ") == 0.0 and parse_hold_margin("-1.5") == -1.5
    for raw in ("abc", "nan", "inf", "21"):
        monkeypatch.setenv("INFERENCE_API_KEY", KEY)
        monkeypatch.setenv("ACTION_HOLD_MARGIN", raw)
        with pytest.raises(ValueError, match="ACTION_HOLD_MARGIN"):
            Settings.from_env()
    monkeypatch.setenv("ACTION_HOLD_MARGIN", "0.75")
    assert Settings.from_env().action_hold_margin == 0.75
