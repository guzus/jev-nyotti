import json

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
from safetensors.torch import load_file

from jev_inference.adapter import load_verified_adapter, sha256_file, verify_exact_tensors, verify_export_config, verify_target_modules
from jev_inference.settings import MODEL_ID, MODEL_REVISION, Settings


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)

    def forward(self, values):
        return self.linear(values)


@pytest.fixture
def exported(tmp_path, monkeypatch):
    config = peft.LoraConfig(r=2, lora_alpha=2, target_modules=["linear"], bias="none")
    trained = peft.get_peft_model(TinyBase(), config)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if "lora_" in name:
                # Values deliberately require more mantissa precision than BF16.
                parameter.data = torch.linspace(0.123456, 0.345678, parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
    trained.save_pretrained(tmp_path, safe_serialization=True, save_embedding_layers=False)
    config_path = tmp_path / "adapter_config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config.update(base_model_name_or_path=MODEL_ID, revision=MODEL_REVISION)
    config_path.write_text(json.dumps(saved_config))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(tmp_path))
    settings = Settings(api_key="test" * 8, adapter_id="guzus/jev-nyotti", adapter_revision="a" * 40,
                        adapter_sha256=sha256_file(tmp_path / "adapter_model.safetensors"))
    return tmp_path, settings


def test_native_peft_reload_preserves_exported_fp32_bytes_and_active_adapter(exported):
    directory, settings = exported
    model, verified = load_verified_adapter(TinyBase(), settings)
    assert verified["exactWeightsVerified"] is True
    assert verified["tensorCount"] == 2
    assert verified["nonzeroPairs"] == 1
    assert all(not parameter.requires_grad for parameter in model.parameters())
    actual = peft.get_peft_model_state_dict(model, save_embedding_layers=False)
    expected = load_file(directory / "adapter_model.safetensors")
    verify_exact_tensors(expected, actual)
    values = torch.ones(1, 4, dtype=torch.bfloat16)
    with torch.inference_mode():
        adapted = model(values)
        with model.disable_adapter():
            base = model(values)
    assert not torch.equal(adapted, base)


def test_wrong_manifest_checksum_fails_before_load(exported):
    _, settings = exported
    wrong = Settings(api_key=settings.api_key, adapter_id=settings.adapter_id,
                     adapter_revision=settings.adapter_revision, adapter_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="checksum"):
        load_verified_adapter(TinyBase(), wrong)


def test_base_revision_mismatch_is_rejected(exported):
    directory, settings = exported
    config = json.loads((directory / "adapter_config.json").read_text())
    config["revision"] = "b" * 40
    with pytest.raises(RuntimeError, match="base revision"):
        verify_export_config(config, settings)


@pytest.mark.parametrize("defect", ["missing", "extra", "dtype", "one_bit"])
def test_tensor_comparison_fails_closed(defect):
    expected = {"weight": torch.tensor([0.123456, 0.345678], dtype=torch.float32)}
    actual = {"weight": expected["weight"].clone()}
    if defect == "missing": actual.clear()
    if defect == "extra": actual["other"] = expected["weight"]
    if defect == "dtype": actual["weight"] = actual["weight"].bfloat16()
    if defect == "one_bit": actual["weight"].view(torch.uint8)[0] ^= 1
    with pytest.raises(RuntimeError):
        verify_exact_tensors(expected, actual)


@pytest.mark.parametrize("defect", ["unknown_target", "shape", "zero", "nonfinite", "missing_pair"])
def test_export_must_match_real_modules_and_contain_nonzero_pairs(defect):
    expected = {
        "base_model.model.linear.lora_A.weight": torch.ones(2, 4),
        "base_model.model.linear.lora_B.weight": torch.ones(3, 2),
    }
    if defect == "unknown_target": expected = {k.replace(".linear.", ".absent."): v for k, v in expected.items()}
    if defect == "shape": expected["base_model.model.linear.lora_A.weight"] = torch.ones(2, 5)
    if defect == "zero": expected["base_model.model.linear.lora_B.weight"].zero_()
    if defect == "nonfinite": expected["base_model.model.linear.lora_A.weight"][0, 0] = float("nan")
    if defect == "missing_pair": expected.pop("base_model.model.linear.lora_B.weight")
    with pytest.raises(RuntimeError):
        verify_target_modules(TinyBase(), expected)
