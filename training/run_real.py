"""Bounded next-hour position imitation pilot; never a production trading model.

The import-safe helpers validate sanitized inputs and calculate action-space metrics.
Only main() imports the GPU stack. Raw examples never enter reports or logs.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ACTIONS = ("long", "short", "flat")
SPLITS = ("train", "validation", "test")
INPUT_FILES = tuple(f"{split}.jsonl" for split in SPLITS) + ("manifest.json",)
TASK = "NEXT_HOUR_POSITION_SIDE"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_LENGTH = 4096
MAX_STEPS = 1024


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", value):
        raise ValueError("invalid artifact identifier")
    return value


def utc_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("cutoff must be UTC with explicit timezone")
    return parsed.timestamp()


def reject_identifier_fields(value):
    forbidden = {"account", "accountid", "userid", "email", "address", "walletaddress",
                 "orderid", "clordid", "execid", "trdmatchid", "uuid", "apikey", "secret"}
    if isinstance(value, dict):
        for key, child in value.items():
            if re.sub(r"[^a-z0-9]", "", key.lower()) in forbidden:
                raise ValueError("identifier or credential field in sanitized job")
            reject_identifier_fields(child)
    elif isinstance(value, list):
        for child in value:
            reject_identifier_fields(child)


def load_dataset(directory):
    """Verify file identities, row schema and chronology before any GPU dispatch."""
    from jev_inference.schemas import Job
    directory = Path(directory)
    for filename in INPUT_FILES:
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing file or disallowed symlink: {filename}")
    manifest = json.loads((directory / "manifest.json").read_text())
    valid_id(manifest["dataset_id"])
    if manifest.get("task") != TASK or manifest.get("model") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("dataset task or pinned model mismatch")
    if not manifest.get("provenance"):
        raise ValueError("dataset provenance is required")
    rows_by_split = {}
    seen = set()
    limits = {"train": 4096, "validation": 128, "test": 128}
    for split in SPLITS:
        filename = f"{split}.jsonl"
        content = (directory / filename).read_bytes()
        if len(content) > 100 * 1024 * 1024:
            raise ValueError("pilot input file exceeds 100 MiB")
        file_info = manifest["files"][filename]
        digest = file_info if isinstance(file_info, str) else file_info["sha256"]
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"SHA256 mismatch: {filename}")
        rows = []
        for line_number, line in enumerate(content.splitlines(), 1):
            # Never echo sensitive record values in validation errors.
            try:
                row = json.loads(line)
                if set(row) != {"job", "target_index", "target_action", "previous_action", "cutoff", "split"}:
                    raise ValueError("row fields")
                reject_identifier_fields(row["job"])
                job = Job.model_validate(row["job"])
                names = [option.name for option in job.options]
                if len(names) != 3 or set(names) != set(ACTIONS):
                    raise ValueError("action options")
                if type(row["target_index"]) is not int or not 0 <= row["target_index"] < 3:
                    raise ValueError("target index")
                if names[row["target_index"]] != row["target_action"] or row["previous_action"] not in ACTIONS:
                    raise ValueError("action mapping")
                if row["split"] != split:
                    raise ValueError("split label")
                timestamp = utc_timestamp(row["cutoff"])
                if timestamp in seen or (rows and timestamp <= utc_timestamp(rows[-1]["cutoff"])):
                    raise ValueError("duplicate or unordered cutoff")
                seen.add(timestamp)
                row["job"] = job
                rows.append(row)
            except Exception:
                raise ValueError(f"invalid sanitized row: {filename}:{line_number}") from None
        count_info = manifest["splits"][split]
        count = count_info if isinstance(count_info, int) else count_info["count"]
        minimum = 4 if split == "train" else 12
        if len(rows) != count or not minimum <= len(rows) <= limits[split]:
            raise ValueError(f"invalid pilot count: {split}")
        rows_by_split[split] = rows
    for older, newer in zip(SPLITS, SPLITS[1:]):
        gap = utc_timestamp(rows_by_split[newer][0]["cutoff"]) - utc_timestamp(rows_by_split[older][-1]["cutoff"])
        if gap < 96 * 3600:
            raise ValueError("chronological splits require at least a 96-hour purge")
    return manifest, rows_by_split


def classification_metrics(targets, predictions):
    if not targets or len(targets) != len(predictions):
        raise ValueError("nonempty aligned actions required")
    confusion = [[0] * 3 for _ in ACTIONS]
    for target, prediction in zip(targets, predictions):
        confusion[ACTIONS.index(target)][ACTIONS.index(prediction)] += 1
    per_class = {}
    for i, action in enumerate(ACTIONS):
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted = sum(row[i] for row in confusion)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[action] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {"count": len(targets), "accuracy": sum(a == b for a, b in zip(targets, predictions)) / len(targets),
            "macro_f1": sum(x["f1"] for x in per_class.values()) / 3,
            "per_class": per_class, "confusion_matrix": confusion,
            "confusion_order": list(ACTIONS), "confusion_axes": "rows=target, columns=prediction"}


def summarize_predictions(examples, results):
    if len(examples) != len(results):
        raise ValueError("evaluation result count mismatch")
    targets = [row["target_action"] for row in examples]
    predictions = [row["job"].options[result["prediction"]].name for row, result in zip(examples, results)]
    previous = [row["previous_action"] for row in examples]
    changed = [i for i, (target, prior) in enumerate(zip(targets, previous)) if target != prior]
    metrics = classification_metrics(targets, predictions)
    metrics["persistence_baseline"] = classification_metrics(targets, previous)
    metrics["position_change_subset"] = {
        "count": len(changed),
        "accuracy": sum(targets[i] == predictions[i] for i in changed) / len(changed) if changed else None,
        "persistence_accuracy": 0.0 if changed else None,
    }
    return metrics


def tensor_hash(tensors):
    """Exact dtype/shape/name/byte fingerprint without leaking tensor content."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main(run_id, dataset_id, deadline_epoch):
    import importlib.metadata
    import random
    import statistics
    import time
    start = time.monotonic()
    out = Path("/artifacts") / valid_id(run_id)
    out.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "dataset_id": valid_id(dataset_id), "task": TASK,
              "model": MODEL_ID, "revision": MODEL_REVISION, "status": "started",
              "reload_status": "not_started", "production_promoted": False}

    def persist():
        report["child_process_elapsed_seconds"] = time.monotonic() - start
        report["gpu_cost_estimate_usd"] = report["child_process_elapsed_seconds"] * .001097
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def emit(event, **fields):
        record = {"event": event, "elapsed_seconds": round(time.monotonic() - start, 3), **fields}
        print(json.dumps(record), flush=True)
        with (out / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    persist()
    try:
        emit("imports_started")
        from unsloth import FastVisionModel  # Must precede transformers/peft.
        import torch
        from transformers import AutoTokenizer
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        from validation import evaluate, compare_tensors, load_exported_adapter
        from data import encode_example
        from jev_inference.labels import select_labels
        from jev_inference.prompt import SYSTEM_PROMPT, format_prompt
        manifest, examples = load_dataset(Path("/inputs") / dataset_id)
        if manifest["dataset_id"] != dataset_id:
            raise ValueError("dataset directory identity mismatch")
        report.update(provenance="sanitized_user_supplied_executions_with_official_historical_candles",
                      dataset_files=manifest["files"],
                      splits={s: {"count": len(examples[s]), "labels": dict(Counter(x["target_action"] for x in examples[s]))} for s in SPLITS},
                      training_method="BF16 LoRA rank16; next-hour position-side imitation; single answer token",
                      requested_steps=MAX_STEPS, batch_size=2, gradient_accumulation=2,
                      cost_note="GPU estimate excludes preparation, startup, CPU/RAM/storage; not an invoice.",
                      limitations=["Position imitation, not profitability or causal trading policy evaluation.",
                                   "Test split is fixed; no test-based tuning or promotion.",
                                   "Reload validation uses training runtime; production compatibility untested."])
        persist()
        torch.manual_seed(3407); random.seed(3407)
        assert torch.cuda.is_available()
        emit("loading", gpu=torch.cuda.get_device_name(0))
        model, _ = FastVisionModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, load_in_4bit=False, dtype=torch.bfloat16,
            use_gradient_checkpointing="unsloth", max_seq_length=MAX_LENGTH,
            trust_remote_code=False, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        probe = format_prompt(tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "{}"}])
        labels = select_labels(tokenizer, probe, count=3)
        encoded = {s: [encode_example(tokenizer, labels, x, MAX_LENGTH) for x in examples[s]] for s in SPLITS}
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407)
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in name and "visual" not in name for name, _ in trainable)
        # Explicit FP32 master adapters preserve save/reload precision.
        for _, parameter in trainable:
            parameter.data = parameter.data.float()
        before = {name: p.detach().cpu().clone() for name, p in trainable}
        report.update(gpu=torch.cuda.get_device_name(0),
                      trainable_parameters=sum(p.numel() for _, p in trainable),
                      min_input_tokens=min(len(x["input_ids"]) for x in encoded["train"]),
                      max_input_tokens=max(len(x["input_ids"]) for x in encoded["train"]),
                      packages={p: importlib.metadata.version(p) for p in ["unsloth", "unsloth_zoo", "torch", "transformers", "peft"]})
        report["baseline"] = {}
        for split in ("validation", "test"):
            result = evaluate(model, encoded[split], labels)
            report["baseline"][split] = summarize_predictions(examples[split], result["rows"])
            persist(); emit("baseline", split=split, metrics=report["baseline"][split])
        FastVisionModel.for_training(model)
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=.01)
        losses, durations = [], []
        indices = list(range(len(encoded["train"]))); random.shuffle(indices)
        cursor = 0
        train_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step in range(MAX_STEPS):
            # Reserve at least 5 minutes of the immutable deadline for fixed evaluations and reload.
            if time.monotonic() - start >= 1200 or time.time() >= deadline_epoch - 330:
                emit("training_time_limit", completed_steps=len(losses)); break
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(2):
                if cursor + 2 > len(indices):
                    random.shuffle(indices); cursor = 0
                batch = [encoded["train"][i] for i in indices[cursor:cursor+2]]; cursor += 2
                width = max(len(x["input_ids"]) for x in batch)
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                ids = torch.full((2, width), pad_id, dtype=torch.long, device="cuda")
                attention = torch.zeros_like(ids); targets = torch.full_like(ids, -100)
                for i, row in enumerate(batch):
                    n = len(row["input_ids"])
                    ids[i, :n] = torch.tensor(row["input_ids"], device="cuda")
                    attention[i, :n] = 1
                    targets[i, :n] = torch.tensor(row["labels"], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(input_ids=ids, attention_mask=attention, labels=targets, use_cache=False)
                    loss = result.loss / 2
                assert torch.isfinite(loss), "non-finite loss"
                loss.backward(); step_loss += loss.item()
                del result, loss, ids, attention, targets
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            assert torch.isfinite(norm), "non-finite gradients"
            optimizer.step(); torch.cuda.synchronize()
            losses.append(step_loss); durations.append(time.monotonic() - tick)
            report.update(completed_steps=len(losses), losses=losses, step_seconds=durations)
            if step == 0 or (step + 1) % 10 == 0:
                persist(); emit("training", step=step + 1, loss=step_loss, step_seconds=durations[-1])
        changed = sum(not torch.equal(before[name], p.detach().cpu()) for name, p in trainable)
        report.update(completed_steps=len(losses), losses=losses, step_seconds=durations,
                      adapter_tensors_changed=changed, training_seconds=time.monotonic() - train_start,
                      training_peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      first_10_mean_loss=statistics.mean(losses[:10]) if losses else None,
                      last_10_mean_loss=statistics.mean(losses[-10:]) if losses else None)
        persist()
        assert len(losses) >= 10, "fewer than10 optimizersteps"
        assert changed > 0, "adapterweights unchanged"
        # Persist export before potentially costly held-out evaluation/reload.
        adapter = out / "adapter"
        model.save_pretrained(str(adapter), safe_serialization=True); tokenizer.save_pretrained(str(adapter))
        config = json.loads((adapter / "adapter_config.json").read_text())
        assert config["base_model_name_or_path"] == MODEL_ID
        config["revision"] = MODEL_REVISION
        (adapter / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        report["adapter_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                   for p in adapter.iterdir() if p.is_file()}
        report["after"] = {}
        parity_before = []
        for split in ("validation", "test"):
            result = evaluate(model, encoded[split], labels)
            report["after"][split] = summarize_predictions(examples[split], result["rows"])
            if split == "validation":
                parity_before = result["rows"][:12]
            persist(); emit("after", split=split, metrics=report["after"][split])
        report["reload_status"] = "pending"; persist()
        saved = load_file(str(adapter / "adapter_model.safetensors"))
        in_memory = get_peft_model_state_dict(model)
        compare_tensors(in_memory, saved)
        report.update(adapter_tensor_dtypes=sorted({str(t.dtype) for t in saved.values()}),
                      memory_tensor_sha256=tensor_hash(in_memory), saved_tensor_sha256=tensor_hash(saved),
                      saved_tensor_bytes_identical=True)
        persist()
        # Release training references before loading another BF16 base.
        del in_memory, optimizer, trainable, before, model
        import gc
        gc.collect(); torch.cuda.empty_cache()
        reloaded = load_exported_adapter(adapter, MAX_LENGTH)
        reloaded_tensors = get_peft_model_state_dict(reloaded)
        compare_tensors(saved, reloaded_tensors)
        report.update(reloaded_tensor_bytes_identical=True, reloaded_tensor_sha256=tensor_hash(reloaded_tensors))
        reloaded_result = evaluate(reloaded, encoded["validation"][:12], labels)
        diff = max(abs(x-y) for a, b in zip(parity_before, reloaded_result["rows"]) for x, y in zip(a["logits"], b["logits"]))
        choices_equal = [x["prediction"] for x in parity_before] == [x["prediction"] for x in reloaded_result["rows"]]
        report.update(reload_cases=12, reload_max_logit_difference=diff, reload_predictions_identical=choices_equal)
        persist()
        assert diff < .15, "reload logit difference exceeds tolerance"
        assert choices_equal, "reload choices differ"
        report.update(reload_status="passed", status="passed"); persist(); emit("complete", status="passed")
    except BaseException as error:
        # Avoid stringifying errors that could include example content or tensor values.
        report.update(status="failed", error_type=type(error).__name__)
        if report["reload_status"] == "pending": report["reload_status"] = "failed"
        persist(); emit("failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
