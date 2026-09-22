"""Fail-closed loading of a pinned, exported FP32 LoRA into native Transformers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .settings import MODEL_ID, Settings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_export_config(config: dict, settings: Settings) -> None:
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise RuntimeError("adapter base model does not match the selected Qwen model")
    if config.get("revision") != settings.revision:
        raise RuntimeError("adapter base revision does not match the pinned serving revision")
    if config.get("peft_type") != "LORA" or config.get("bias") != "none" or config.get("modules_to_save"):
        raise RuntimeError("only LoRA-only checkpoints with no saved base modules are supported")


def verify_target_modules(base: Any, expected: dict) -> int:
    """Check every exported A/B pair against a real base module before PEFT loads.

    Unsloth's exported keys already match the selected native Transformers
    architecture. Do not guess or discard prefixes when an architecture changes.
    """
    import torch

    pairs: dict[str, dict] = {}
    prefix = "base_model.model."
    for name, tensor in expected.items():
        if not name.startswith(prefix):
            raise RuntimeError("adapter has an unsupported tensor prefix")
        module_name, separator, suffix = name[len(prefix):].rpartition(".lora_")
        if not separator or suffix not in {"A.weight", "B.weight"}:
            raise RuntimeError("adapter contains tensors outside standard LoRA A/B weights")
        if tensor.dtype != torch.float32 or tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise RuntimeError("adapter tensors must be finite FP32 matrices")
        pairs.setdefault(module_name, {})[suffix[0]] = tensor
    if not pairs:
        raise RuntimeError("adapter contains no LoRA tensors")
    nonzero_pairs = 0
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise RuntimeError("adapter is missing a LoRA A/B pair")
        try:
            module = base.get_submodule(name)
        except AttributeError as exc:
            raise RuntimeError("adapter target does not exist in the serving architecture") from exc
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError("adapter target is not a native linear layer")
        a, b = pair["A"], pair["B"]
        if a.shape[1] != module.in_features or b.shape[0] != module.out_features or a.shape[0] != b.shape[1]:
            raise RuntimeError("adapter tensor shape does not match the serving architecture")
        if torch.count_nonzero(a).item() and torch.count_nonzero(b).item():
            nonzero_pairs += 1
    if not nonzero_pairs:
        raise RuntimeError("adapter has no nonzero LoRA updates")
    return nonzero_pairs


def verify_exact_tensors(expected: dict, actual: dict) -> None:
    """Detect missing/extra keys, dtype narrowing, and even one changed saved bit."""
    import torch

    if expected.keys() != actual.keys():
        raise RuntimeError("loaded adapter tensor keys differ from the exported checkpoint")
    for name, tensor in expected.items():
        other = actual[name].detach().cpu().contiguous()
        tensor = tensor.detach().cpu().contiguous()
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            raise RuntimeError("loaded adapter tensor shape or dtype differs from export")
        if not torch.equal(tensor.view(torch.uint8), other.view(torch.uint8)):
            raise RuntimeError("loaded adapter tensor bytes differ from export")


def load_verified_adapter(base: Any, settings: Settings) -> tuple[Any, dict]:
    from huggingface_hub import snapshot_download
    from peft import PeftModel, get_model_status, get_peft_model_state_dict
    from safetensors.torch import load_file

    if not settings.adapter_id or not settings.adapter_revision:
        raise RuntimeError("a pinned adapter must be configured")
    directory = Path(snapshot_download(
        settings.adapter_id,
        revision=settings.adapter_revision,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    ))
    verify_export_config(json.loads((directory / "adapter_config.json").read_text()), settings)
    weights_path = directory / "adapter_model.safetensors"
    digest = sha256_file(weights_path)
    if settings.adapter_sha256 and digest != settings.adapter_sha256:
        raise RuntimeError("adapter file checksum differs from the deployment manifest")
    expected = load_file(str(weights_path), device="cpu")
    nonzero_pairs = verify_target_modules(base, expected)

    model = PeftModel.from_pretrained(
        base,
        str(directory),
        is_trainable=False,
        # Promote adapter parameters before FP32 checkpoint assignment. Never
        # narrow the export to BF16 and then widen the already-rounded values.
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=False,
        key_mapping={},  # The verified export already has the exact native keys.
        local_files_only=True,
    )
    actual = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    verify_exact_tensors(expected, actual)
    model.set_adapter("default", inference_mode=True)
    model.requires_grad_(False)
    model.eval()
    status = get_model_status(model)
    if status.enabled is not True or status.active_adapters != ["default"] or status.merged_adapters:
        raise RuntimeError("verified adapter is not active in inference mode")
    if status.num_adapter_layers != len(expected) // 2:
        raise RuntimeError("active adapter layer count differs from export")
    return model, {
        "id": settings.adapter_id,
        "revision": settings.adapter_revision,
        "sha256": digest,
        "tensorCount": len(expected),
        "dtype": "float32",
        "nonzeroPairs": nonzero_pairs,
        "exactWeightsVerified": True,
        "active": True,
    }
