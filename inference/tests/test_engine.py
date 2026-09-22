from contextlib import nullcontext
import math
import sys
from types import SimpleNamespace

import pytest

from jev_inference.engine import PreparedJob, QwenEngine
from jev_inference.settings import Settings


def fake_torch(monkeypatch):
    torch = SimpleNamespace(
        long="long",
        inference_mode=nullcontext,
        tensor=lambda data, **_kwargs: data,
        ones_like=lambda data: [[1] * len(data[0])],
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


class SelectedTensor:
    def __init__(self, values):
        self.values = values

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def test_one_forward_per_question_gathers_all_requested_vocab_ids(monkeypatch):
    fake_torch(monkeypatch)
    calls = []
    gathered = []

    class Logits:
        def __getitem__(self, selection):
            assert selection[:2] == (0, -1)
            gathered.append(selection[2])
            # The last candidate is deliberately a very low-logit token, not top-k.
            return SelectedTensor([-float(token_id) for token_id in selection[2]])

    def forward(**kwargs):
        calls.append(kwargs)
        assert kwargs["logits_to_keep"] == 1
        assert kwargs["use_cache"] is False
        return SimpleNamespace(logits=Logits())

    engine = QwenEngine(Settings(api_key="a" * 32))
    engine.model = forward
    options = [23, 8, 900_001] + list(range(100, 352))
    prepared = [PreparedJob([1, 2, 3], options), PreparedJob([7, 8], [89])]
    result = engine.score(prepared)
    assert len(calls) == 2
    assert calls[0]["input_ids"] == [[1, 2, 3]]
    assert calls[1]["input_ids"] == [[7, 8]]
    assert gathered == [options, [89]]
    assert len(result[0].logits) == 255
    assert result[0].logits[2] == -900_001.0
    assert result[1].inputTokens == 2


def test_nonfinite_logits_fail_instead_of_fabricating_probabilities(monkeypatch):
    fake_torch(monkeypatch)

    class Logits:
        def __getitem__(self, _selection):
            return SelectedTensor([math.nan])

    engine = QwenEngine(Settings(api_key="a" * 32))
    engine.model = lambda **_kwargs: SimpleNamespace(logits=Logits())
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.score([PreparedJob([1], [2])])
