from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time

import pytest
from fastapi.testclient import TestClient

from jev_inference.app import create_app
from jev_inference.engine import InputTooLong
from jev_inference.schemas import JobScore
from jev_inference.settings import MODEL_ID, MODEL_REVISION, Settings

KEY = "test-only-key-012345678901234567890123456789"
HEADERS = {"Authorization": f"Bearer {KEY}"}
BODY = {"jobs": [{"state": {"position": "flat"}, "instructions": "Classify.", "options": [{"name": "LONG"}, {"name": "HOLD"}]}]}


class FakeEngine:
    """Explicit test dependency; never imported by production modules."""

    def __init__(self):
        self.loaded = False
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.delay = 0

    def load(self):
        self.loaded = True

    def prepare(self, jobs):
        assert self.loaded
        if jobs[0].state == "too_long":
            raise InputTooLong()
        if jobs[0].state == "error":
            raise RuntimeError(f"do not disclose {KEY}")
        return jobs

    def score(self, jobs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls += 1
        time.sleep(self.delay)
        result = [JobScore(logits=[float(i) for i in range(len(job.options))], inputTokens=17 + j) for j, job in enumerate(jobs)]
        self.active -= 1
        return result


@pytest.fixture
def running():
    engine = FakeEngine()
    app = create_app(Settings(api_key=KEY), engine_factory=lambda: engine)
    with TestClient(app) as client:
        yield client, engine


def test_missing_credentials_fail_before_engine_creation():
    with pytest.raises(ValueError, match="INFERENCE_API_KEY"):
        Settings(api_key="short")


@pytest.mark.parametrize("key", ["a" * 31, " " + "a" * 32, "한" * 32])
def test_invalid_credentials_rejected(key):
    with pytest.raises(ValueError):
        Settings(api_key=key)


def test_adapter_revision_must_be_pinned_and_both_fields_present():
    with pytest.raises(ValueError):
        Settings(api_key=KEY, adapter_id="owner/adapter")
    with pytest.raises(ValueError):
        Settings(api_key=KEY, adapter_id="owner/adapter", adapter_revision="main")
    config = Settings(api_key=KEY, adapter_id="owner/adapter", adapter_revision="a" * 40)
    assert config.provenance_revision.endswith("+lora:owner/adapter@" + "a" * 40)


def test_readiness_reflects_actual_load_and_base_provenance(running):
    client, engine = running
    response = client.get("/healthz")
    assert response.status_code == 200
    assert engine.loaded
    assert response.json()["ready"] is True
    assert response.json()["model"] == MODEL_ID
    assert response.json()["revision"] == MODEL_REVISION
    assert response.json()["fineTuned"] is False


def test_health_is_unready_without_startup():
    app = create_app(Settings(api_key=KEY), engine_factory=FakeEngine)
    client = TestClient(app)
    assert client.get("/healthz").status_code == 503
    assert client.post("/score", headers=HEADERS, json=BODY).status_code == 503


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer invalid"}, {"Authorization": "Basic anything"}])
def test_requests_require_auth_before_inference(running, headers):
    client, engine = running
    response = client.post("/score", headers=headers, json=BODY)
    assert response.status_code == 401
    assert engine.calls == 0
    assert KEY not in response.text


def test_response_preserves_every_option_in_order_including_255(running):
    client, engine = running
    body = copy.deepcopy(BODY)
    body["jobs"].append({"state": ["same context"], "instructions": {"rule": "classify"}, "options": [{"name": f"option-{i}"} for i in range(255)]})
    response = client.post("/score", headers=HEADERS, json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["scores"][0]["logits"] == [0.0, 1.0]
    assert result["scores"][1]["logits"] == [float(i) for i in range(255)]
    assert result["scores"][1]["inputTokens"] == 18
    assert result["elapsedMs"] >= 0
    assert engine.calls == 1


@pytest.mark.parametrize("mutate", [
    lambda body: body.update(jobs=[]),
    lambda body: body.update(jobs=body["jobs"] * 9),
    lambda body: body["jobs"][0].update(state=123),
    lambda body: body["jobs"][0].update(state=None),
    lambda body: body["jobs"][0].update(instructions=True),
    lambda body: body["jobs"][0].update(question_id="hidden-question"),
    lambda body: body["jobs"][0].update(options=[]),
    lambda body: body["jobs"][0].update(options=[{"name": f"{i}"} for i in range(256)]),
    lambda body: body["jobs"][0].update(options=[{"name": "same"}, {"name": "same"}]),
])
def test_invalid_shape_is_rejected_without_gpu_work(running, mutate):
    client, engine = running
    body = copy.deepcopy(BODY)
    mutate(body)
    response = client.post("/score", headers=HEADERS, json=body)
    assert response.status_code == 422
    assert engine.calls == 0


@pytest.mark.parametrize("content", [b"{", b"\xff", b'{"jobs":[{"state":{"a":NaN}}]}', b'{"jobs":[{"state":{"a":1e400},"instructions":"x","options":[{"name":"x"}]}]}'])
def test_malformed_or_nonfinite_json_is_rejected(running, content):
    client, engine = running
    response = client.post("/score", headers=HEADERS, content=content)
    assert response.status_code == 422
    assert engine.calls == 0


def test_chunked_body_is_bounded_before_parsing(running):
    client, engine = running
    response = client.post("/score", headers=HEADERS, content=iter([b"x" * 32768, b"x" * 32769]))
    assert response.status_code == 413
    assert engine.calls == 0


def test_context_is_not_silently_truncated_and_exception_details_are_private(running):
    client, engine = running
    body = copy.deepcopy(BODY)
    body["jobs"][0]["state"] = "too_long"
    assert client.post("/score", headers=HEADERS, json=body).json()["error"]["code"] == "input_too_long"
    body["jobs"][0]["state"] = "error"
    response = client.post("/score", headers=HEADERS, json=body)
    assert response.status_code == 503
    assert KEY not in response.text
    assert engine.calls == 0


def test_gpu_work_is_serialized_and_queue_is_bounded(running):
    client, engine = running
    engine.delay = 0.1
    barrier = threading.Barrier(12)

    def request():
        barrier.wait()
        return client.post("/score", headers=HEADERS, json=BODY).status_code

    with ThreadPoolExecutor(max_workers=12) as pool:
        statuses = list(pool.map(lambda _: request(), range(12)))
    assert set(statuses) <= {200, 429}
    assert 200 in statuses and 429 in statuses
    assert engine.max_active == 1
