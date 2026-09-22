"""Recheck an existing rehearsal export without spending GPU time on training."""
import json
from pathlib import Path
import sys
import time

START = time.monotonic()
RUN_ID, SOURCE_ID = sys.argv[1:3]
OUT = Path("/artifacts") / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
source = Path("/artifacts") / SOURCE_ID
original = json.loads((source / "report.json").read_text())
assert original["provenance"] == "synthetic_rehearsal_not_trader_data"
assert original["saved_tensor_bytes_identical"] is True
report = {"run_id": RUN_ID, "source_run_id": SOURCE_ID,
          "provenance": original["provenance"], "mode": "verify_existing_adapter",
          "reload_status": "pending"}


def persist():
    elapsed = time.monotonic() - START
    report.update(child_process_elapsed_seconds=elapsed, gpu_cost_estimate_usd=elapsed * 0.001097,
                  cost_note="GPU estimate for this verification process only; excludes source training runs, startup, CPU/RAM/storage. Not an invoice.")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


try:
    from unsloth import FastVisionModel
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from data import encode_example, synthetic_examples
    from validation import compare_tensors, evaluate, load_exported_adapter
    from jev_inference.labels import select_labels
    from jev_inference.prompt import SYSTEM_PROMPT, format_prompt
    from jev_inference.settings import MODEL_ID, MODEL_REVISION

    assert original["model"] == MODEL_ID and original["revision"] == MODEL_REVISION
    adapter = source / "adapter"
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == MODEL_ID and config["revision"] == MODEL_REVISION
    saved = load_file(str(adapter / "adapter_model.safetensors"))
    model = load_exported_adapter(adapter)
    compare_tensors(saved, get_peft_model_state_dict(model))
    report["reloaded_tensor_bytes_identical"] = True
    print(json.dumps({"event": "adapter_bytes_verified", "elapsed_seconds": time.monotonic()-START}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    probe = format_prompt(tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "{}"}])
    labels = select_labels(tokenizer, probe, count=3)
    rows = [encode_example(tokenizer, labels, x, 4096) for x in synthetic_examples(12, 99173)]
    result = evaluate(model, rows, labels)
    prior = original["evaluation_after"]
    assert [x["target"] for x in result["rows"]] == [x["target"] for x in prior["rows"]]
    difference = max(abs(a-b) for x,y in zip(prior["rows"], result["rows"])
                     for a,b in zip(x["logits"], y["logits"]))
    identical = [x["prediction"] for x in prior["rows"]] == [x["prediction"] for x in result["rows"]]
    report.update(evaluation_reloaded=result, reload_max_logit_difference=difference,
                  reload_predictions_identical=identical, model=MODEL_ID, revision=MODEL_REVISION,
                  gpu=torch.cuda.get_device_name(0))
    assert difference < 0.15, f"reload logit difference too large: {difference}"
    assert identical, "reload predictions differ"
    report["reload_status"] = "passed"
except Exception as error:
    report.update(reload_status="failed", reload_error=f"{type(error).__name__}: {error}")
    raise
finally:
    persist()
print(json.dumps(report), flush=True)
