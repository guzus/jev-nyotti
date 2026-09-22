"""Same-runtime tensor and output parity checks for synthetic LoRA exports."""
from unsloth import FastVisionModel
import torch
from peft import PeftModel
from jev_inference.settings import MODEL_ID, MODEL_REVISION


def evaluate(current, evaluation, labels):
    FastVisionModel.for_inference(current)
    results = []
    with torch.inference_mode():
        for row in evaluation:
            ids = torch.tensor([row["input_ids"][:-1]], device="cuda")
            result = current(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False, logits_to_keep=1)
            scores = result.logits[0, -1, [x.token_id for x in labels]].float()
            assert torch.isfinite(scores).all()
            results.append({"target": row["target_index"], "prediction": scores.argmax().item(),
                            "logits": scores.cpu().tolist()})
    return {"accuracy": sum(x["target"] == x["prediction"] for x in results) / len(results),
            "rows": results}


def compare_tensors(expected, actual):
    assert expected.keys() == actual.keys(), "adapter tensor keys differ"
    for name, tensor in expected.items():
        other = actual[name].detach().cpu().contiguous()
        tensor = tensor.detach().cpu().contiguous()
        assert tensor.shape == other.shape, f"adapter shape differs: {name}"
        assert tensor.dtype == other.dtype, f"adapter dtype differs: {name}"
        assert torch.equal(tensor.view(torch.uint8), other.view(torch.uint8)), f"adapter bytes differ: {name}"


def load_exported_adapter(adapter, max_length=4096):
    # PEFT creates adapter parameters in the BF16 base dtype. Its default True
    # promotes them to F32 BEFORE loading our F32 checkpoint, preserving every bit.
    fresh_base, _ = FastVisionModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, load_in_4bit=False, dtype=torch.bfloat16,
        use_gradient_checkpointing="unsloth", max_seq_length=max_length,
        trust_remote_code=False, local_files_only=True,
    )
    reloaded = PeftModel.from_pretrained(
        fresh_base, str(adapter), is_trainable=True, autocast_adapter_dtype=True,
        local_files_only=True,
    )
    return FastVisionModel.post_patch_model(reloaded, "unsloth", trust_remote_code=False)
