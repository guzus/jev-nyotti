from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

MODEL_ID = "Qwen/Qwen3.5-4B"
# Official Hugging Face model revision, checked 2026-09-22. No moving main branch.
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_BODY_BYTES = 65_536
MAX_INPUT_TOKENS = 8_192
MAX_JOBS = 8
MAX_OPTIONS = 255
MAX_HOLD_MARGIN = 20.0
# /action policy: 'lora' scores Qwen (+ optional adapter); 'numeric' uses the pinned pure-Python
# numeric artifact (numeric_policy.py) and never loads Qwen.
ACTION_POLICIES = ("lora", "numeric")


def parse_hold_margin(raw: str | None):
    """ACTION_HOLD_MARGIN env value: a number, or JSON {"flat": m, "position": m}; unset means 0.0."""
    if raw is None or not raw.strip():
        return 0.0
    try:
        if raw.strip().startswith("{"):
            return {k: float(v) for k, v in json.loads(raw).items()}
        return float(raw)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("ACTION_HOLD_MARGIN must be a finite number with |value| <= 20") from None


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    device: str = "cuda"
    revision: str = MODEL_REVISION
    adapter_id: str | None = None
    adapter_revision: str | None = None
    adapter_sha256: str | None = None
    # ACTION_V1 decision rule: subtract this from the hold logit before argmax.
    action_hold_margin: float | dict = 0.0
    action_policy: str = "lora"
    numeric_model_path: str | None = None
    numeric_model_sha256: str | None = None

    def __post_init__(self) -> None:
        # Validate before model downloads or GPU work. Never include supplied values.
        if len(self.api_key) < 32 or not self.api_key.isascii() or any(c.isspace() for c in self.api_key):
            raise ValueError("INFERENCE_API_KEY must contain at least 32 non-whitespace ASCII characters")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("INFERENCE_DEVICE must be cuda or cpu")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("MODEL_REVISION must be a pinned 40-character Hugging Face commit SHA")
        if bool(self.adapter_id) != bool(self.adapter_revision):
            raise ValueError("LORA_MODEL_ID and LORA_REVISION must be configured together")
        if self.adapter_id:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.adapter_id):
                raise ValueError("LORA_MODEL_ID must be a Hugging Face owner/repository")
            if not re.fullmatch(r"[0-9a-f]{40}", self.adapter_revision or ""):
                raise ValueError("LORA_REVISION must be a pinned 40-character commit SHA")
        if self.adapter_sha256:
            if not self.adapter_id or not re.fullmatch(r"[0-9a-f]{64}", self.adapter_sha256):
                raise ValueError("LORA_SHA256 requires a configured adapter and its 64-character SHA-256")
        from .action_task import check_margin
        try:
            check_margin(self.action_hold_margin, MAX_HOLD_MARGIN)
        except ValueError:
            raise ValueError("ACTION_HOLD_MARGIN must be a finite number with |value| <= 20") from None
        if self.action_policy not in ACTION_POLICIES:
            raise ValueError("ACTION_POLICY must be lora or numeric")
        if self.action_policy == "numeric":
            if not self.numeric_model_path or not re.fullmatch(r"[0-9a-f]{64}", self.numeric_model_sha256 or ""):
                raise ValueError("numeric ACTION_POLICY requires NUMERIC_MODEL_PATH and a 64-character NUMERIC_MODEL_SHA256")
        elif self.numeric_model_path or self.numeric_model_sha256:
            raise ValueError("NUMERIC_MODEL_* is only valid with ACTION_POLICY=numeric")

    @property
    def provenance_revision(self) -> str:
        if self.adapter_id:
            return f"{self.revision}+lora:{self.adapter_id}@{self.adapter_revision}"
        return self.revision

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.environ.get("INFERENCE_API_KEY", ""),
            device=os.environ.get("INFERENCE_DEVICE", "cuda"),
            revision=os.environ.get("MODEL_REVISION", MODEL_REVISION),
            adapter_id=os.environ.get("LORA_MODEL_ID") or None,
            adapter_revision=os.environ.get("LORA_REVISION") or None,
            adapter_sha256=os.environ.get("LORA_SHA256") or None,
            action_hold_margin=parse_hold_margin(os.environ.get("ACTION_HOLD_MARGIN")),
            action_policy=os.environ.get("ACTION_POLICY") or "lora",
            numeric_model_path=os.environ.get("NUMERIC_MODEL_PATH") or None,
            numeric_model_sha256=os.environ.get("NUMERIC_MODEL_SHA256") or None,
        )
