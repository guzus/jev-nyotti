"""Train a small LoRA and prove save/reload on synthetic classifier examples."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys
import time

START = time.monotonic()
RUN_ID = sys.argv[1]
OUT = Path("/artifacts") / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)


def emit(event, **fields):
    record = {"event": event, "elapsed_seconds": round(time.monotonic() - START, 3), **fields}
    print(json.dumps(record), flush=True)
    with (OUT / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


emit("imports_started", provenance="synthetic_rehearsal")
from unsloth import FastVisionModel  # Must precede transformers / peft imports.
import torch
from transformers import AutoTokenizer
from peft import get_peft_model_state_dict
from validation import evaluate, compare_tensors, load_exported_adapter
from safetensors.torch import load_file
from data import encode_example, synthetic_examples
from jev_inference.labels import select_labels
from jev_inference.prompt import SYSTEM_PROMPT, format_prompt
from jev_inference.settings import MODEL_ID, MODEL_REVISION

torch.manual_seed(3407)
random.seed(3407)
MAX_LENGTH = 4096
STEPS = 100
BATCH_SIZE = 2
ACCUMULATION = 2
assert torch.cuda.is_available()
emit("loading", gpu=torch.cuda.get_device_name(0))
model, processor = FastVisionModel.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, load_in_4bit=False, dtype=torch.bfloat16,
    use_gradient_checkpointing="unsloth", max_seq_length=MAX_LENGTH,
    trust_remote_code=False, local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
probe = format_prompt(tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "{}"}])
labels = select_labels(tokenizer, probe, count=3)
train = [encode_example(tokenizer, labels, x, MAX_LENGTH) for x in synthetic_examples(192, 3407)]
evaluation = [encode_example(tokenizer, labels, x, MAX_LENGTH) for x in synthetic_examples(12, 99173)]
assert not {tuple(x["input_ids"]) for x in train} & {tuple(x["input_ids"]) for x in evaluation}
model = FastVisionModel.get_peft_model(
    model, finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
)
trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
assert trainable and all("lora_" in name for name, _ in trainable)
assert not any("visual" in name for name, _ in trainable)
before = {name: p.detach().float().cpu().clone() for name, p in trainable}
emit("model_ready", trainable_parameters=sum(p.numel() for _, p in trainable),
     examples=len(train), min_tokens=min(len(x["input_ids"]) for x in train),
     max_tokens=max(len(x["input_ids"]) for x in train), labels=[x.text for x in labels])


baseline = evaluate(model, evaluation, labels)
emit("baseline", synthetic_rule_accuracy=baseline["accuracy"])
FastVisionModel.for_training(model)
optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=0.01)
losses, durations, tokens_per_step = [], [], []
indices = list(range(len(train)))
random.shuffle(indices)
cursor = 0
train_start = time.monotonic()
torch.cuda.reset_peak_memory_stats()
for step in range(STEPS):
    if time.monotonic() - START > 1000:
        emit("training_time_limit", completed_steps=step)
        break
    torch.cuda.synchronize()
    tick = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0
    token_count = 0
    for _ in range(ACCUMULATION):
        if cursor + BATCH_SIZE > len(indices):
            random.shuffle(indices)
            cursor = 0
        rows = [train[i] for i in indices[cursor:cursor+BATCH_SIZE]]
        cursor += BATCH_SIZE
        width = max(len(x["input_ids"]) for x in rows)
        input_ids = torch.full((BATCH_SIZE, width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
        attention = torch.zeros_like(input_ids)
        targets = torch.full_like(input_ids, -100)
        for i, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[i, :length] = torch.tensor(row["input_ids"], device="cuda")
            attention[i, :length] = 1
            targets[i, :length] = torch.tensor(row["labels"], device="cuda")
            token_count += length
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(input_ids=input_ids, attention_mask=attention, labels=targets, use_cache=False)
            loss = result.loss / ACCUMULATION
        assert torch.isfinite(loss), "non-finite training loss"
        loss.backward()
        step_loss += loss.item()
        del result, loss, input_ids, attention, targets
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
    assert torch.isfinite(norm), "non-finite gradients"
    optimizer.step()
    torch.cuda.synchronize()
    durations.append(time.monotonic() - tick)
    losses.append(step_loss)
    tokens_per_step.append(token_count)
    if step == 0 or (step + 1) % 10 == 0:
        emit("training", step=step+1, loss=step_loss, step_seconds=durations[-1],
             input_tokens=token_count)
train_seconds = time.monotonic() - train_start
assert len(losses) >= 10, "too few training steps completed"
changed = sum(not torch.equal(before[name], p.detach().float().cpu()) for name, p in trainable)
assert changed > 0, "adapter weights did not change"
after = evaluate(model, evaluation, labels)
adapter = OUT / "adapter"
model.save_pretrained(str(adapter), safe_serialization=True)
tokenizer.save_pretrained(str(adapter))
config = json.loads((adapter / "adapter_config.json").read_text())
assert config["base_model_name_or_path"] == MODEL_ID
# Record the exact official base revision, including for subsequent PEFT loading.
config["revision"] = MODEL_REVISION
(adapter / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in adapter.iterdir() if p.is_file()}
assert files.get("adapter_model.safetensors", {}).get("bytes", 0) > 0
emit("adapter_saved", changed_tensors=changed, synthetic_rule_accuracy=after["accuracy"])
measured_seconds = sum(durations[5:])
measured_tokens = sum(tokens_per_step[5:])
elapsed = time.monotonic() - START
report = {
    "run_id": RUN_ID, "provenance": "synthetic_rehearsal_not_trader_data",
    "model": MODEL_ID, "revision": MODEL_REVISION,
    "gpu": torch.cuda.get_device_name(0), "training_method": "BF16 LoRA rank 16; text-only",
    "completed_steps": len(losses), "requested_steps": STEPS,
    "batch_size": BATCH_SIZE, "gradient_accumulation": ACCUMULATION,
    "train_examples": len(train), "evaluation_examples": len(evaluation),
    "min_input_tokens": min(len(x["input_ids"]) for x in train),
    "mean_input_tokens": statistics.mean(len(x["input_ids"]) for x in train),
    "max_input_tokens": max(len(x["input_ids"]) for x in train),
    "training_seconds_including_compile": train_seconds,
    "steady_tokens_per_second_after_5_steps": measured_tokens / measured_seconds,
    "median_step_seconds_after_5_steps": statistics.median(durations[5:]),
    "first_10_mean_loss": statistics.mean(losses[:10]), "last_10_mean_loss": statistics.mean(losses[-10:]),
    "synthetic_rule_accuracy_before": baseline["accuracy"], "synthetic_rule_accuracy_after": after["accuracy"],
    "adapter_tensors_changed": changed, "reload_status": "pending",
    "training_peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
    "child_process_elapsed_seconds": elapsed,
    "gpu_cost_estimate_usd": elapsed * 0.001097,
    "cost_note": "Estimate for observed child runtime only; excludes container startup, CPU/RAM/storage and CPU preparation. Not an invoice.",
    "packages": {p: importlib.metadata.version(p) for p in ["unsloth", "unsloth_zoo", "torch", "transformers", "peft", "trl"]},
    "adapter_files": files,
    "losses": losses, "step_seconds": durations, "input_tokens_per_step": tokens_per_step,
    "evaluation_before": baseline, "evaluation_after": after,
    "limitations": ["Synthetic rule task only; no trader records or profitability evaluation.",
                    "Save/reload checked within Unsloth environment; live inference adapter compatibility not yet tested.",
                    "No production model or endpoint was modified."],
}


def persist_report():
    elapsed = time.monotonic() - START
    report["child_process_elapsed_seconds"] = elapsed
    report["gpu_cost_estimate_usd"] = elapsed * 0.001097
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


# Save measurements even if subsequent serialization validation fails.
persist_report()
try:
    saved = load_file(str(adapter / "adapter_model.safetensors"))
    compare_tensors(get_peft_model_state_dict(model), saved)
    report["adapter_tensor_dtypes"] = sorted({str(t.dtype) for t in saved.values()})
    report["saved_tensor_bytes_identical"] = True
    persist_report()
    reloaded = load_exported_adapter(adapter, MAX_LENGTH)
    compare_tensors(saved, get_peft_model_state_dict(reloaded))
    report["reloaded_tensor_bytes_identical"] = True
    reloaded_result = evaluate(reloaded, evaluation, labels)
    max_difference = max(abs(x-y) for a,b in zip(after["rows"], reloaded_result["rows"])
                         for x,y in zip(a["logits"],b["logits"]))
    predictions_equal = ([x["prediction"] for x in after["rows"]]
                         == [x["prediction"] for x in reloaded_result["rows"]])
    report.update(reload_max_logit_difference=max_difference,
                  reload_predictions_identical=predictions_equal,
                  evaluation_reloaded=reloaded_result)
    persist_report()
    assert max_difference < 0.15, f"reload logit difference too large: {max_difference}"
    assert predictions_equal, "reload predictions differ"
    report["reload_status"] = "passed"
except Exception as error:
    report["reload_status"] = "failed"
    report["reload_error"] = f"{type(error).__name__}: {error}"
    persist_report()
    raise
persist_report()
emit("complete", report=report)
