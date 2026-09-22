from __future__ import annotations

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


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    device: str = "cuda"
    revision: str = MODEL_REVISION
    adapter_id: str | None = None
    adapter_revision: str | None = None
    adapter_sha256: str | None = None

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
        )
