from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from .labels import Label, select_labels
from .prompt import SYSTEM_PROMPT, format_prompt, messages_for_job
from .schemas import Job, JobScore
from .settings import MAX_INPUT_TOKENS, MODEL_ID, Settings


class InputTooLong(ValueError):
    pass


@dataclass(frozen=True)
class PreparedJob:
    input_ids: list[int]
    candidate_ids: list[int]


class Engine(Protocol):
    def load(self) -> None: ...

    def prepare(self, jobs: list[Job]) -> list[PreparedJob]: ...

    def score(self, prepared: list[PreparedJob]) -> list[JobScore]: ...


class QwenEngine:
    """One model per process. The HTTP service serializes calls into this engine."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer: Any = None
        self.model: Any = None
        self.labels: list[Label] = []

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no compatible GPU was found")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=self.settings.revision, trust_remote_code=False,
        )
        probe = format_prompt(self.tokenizer, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "{}"},
        ])
        self.labels = select_labels(self.tokenizer, probe)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=self.settings.revision,
            dtype=torch.bfloat16 if self.settings.device == "cuda" else torch.float32,
            device_map=self.settings.device,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        if self.settings.adapter_id:
            from peft import PeftConfig, PeftModel

            adapter_config = PeftConfig.from_pretrained(
                self.settings.adapter_id, revision=self.settings.adapter_revision,
            )
            if adapter_config.base_model_name_or_path != MODEL_ID:
                raise RuntimeError("adapter base_model_name_or_path must match the selected Qwen model")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.settings.adapter_id,
                revision=self.settings.adapter_revision,
                is_trainable=False,
            )
        self.model.eval()

    def prepare(self, jobs: list[Job]) -> list[PreparedJob]:
        if self.model is None:
            raise RuntimeError("model not loaded")
        prepared = []
        # Validate the entire request before any GPU forward pass. Never truncate.
        for job in jobs:
            prompt = format_prompt(self.tokenizer, messages_for_job(job, self.labels))
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) > MAX_INPUT_TOKENS:
                raise InputTooLong(f"each formatted job must fit within {MAX_INPUT_TOKENS} input tokens")
            labels = self.labels[:len(job.options)]
            # Check the boundary against this actual prompt, not only the startup probe.
            for label in labels:
                if self.tokenizer.encode(prompt + label.text, add_special_tokens=False) != token_ids + [label.token_id]:
                    raise RuntimeError("label token boundary changed")
            prepared.append(PreparedJob(input_ids=token_ids, candidate_ids=[label.token_id for label in labels]))
        return prepared

    def score(self, prepared: list[PreparedJob]) -> list[JobScore]:
        import torch

        scores = []
        with torch.inference_mode():
            for job in prepared:
                inputs = torch.tensor([job.input_ids], dtype=torch.long, device=self.settings.device)
                # One forward pass per independent question. Compute only the final
                # position, but retain the full vocabulary before gathering ALL labels.
                output = self.model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
                selected = output.logits[0, -1, job.candidate_ids].float().cpu().tolist()
                if not all(math.isfinite(value) for value in selected):
                    raise RuntimeError("model produced non-finite logits")
                scores.append(JobScore(logits=selected, inputTokens=len(job.input_ids)))
                del output, inputs
        return scores
