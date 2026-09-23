"""ACTION_V1 serving helpers shared by POST /action and the historical action replay.

The prompt is built only by `action_task.build_job`; this module adds request parsing,
extra fail-closed guards, and the frozen decision rule (`argmax(logits + b)` where `b`
subtracts the configured margin from `hold` only).
"""
from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, model_validator

from . import action_task
from .schemas import Job, StrictModel

Number = StrictInt | StrictFloat
ISO_Z = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
# Tolerate small gateway/GPU clock skew; a later cutoff cannot have closed candles yet.
FUTURE_SKEW_SECONDS = 60


class ActionCandle(StrictModel):
    time: Annotated[StrictInt, Field(ge=0)]
    open: Number
    high: Number
    low: Number
    close: Number
    volume: Number


class ActionPosition(StrictModel):
    side: Literal["flat", "long", "short"]
    entry_price: Number | None
    opened_at: Annotated[StrictInt, Field(ge=0)] | None
    last_trade_at: Annotated[StrictInt, Field(ge=0)] | None


class ActionRequest(StrictModel):
    market: Annotated[str, Field(min_length=1, max_length=120)]
    cutoff: Annotated[str, Field(min_length=20, max_length=20)]
    candles: Annotated[list[ActionCandle], Field(min_length=action_task.LOOKBACK, max_length=action_task.LOOKBACK)]
    position: ActionPosition

    @model_validator(mode="after")
    def nonblank_market(self) -> "ActionRequest":
        if not self.market.strip():
            raise ValueError("market must not be blank")
        return self


class ActionOption(StrictModel):
    name: str
    probability: float


class ActionResponse(StrictModel):
    model: str
    policy: Literal["lora", "numeric"]
    revision: str
    task: Literal["ACTION_V1"]
    action: str
    options: list[ActionOption]
    holdMargin: float
    inputTokens: Annotated[int, Field(ge=0)]  # 0 for the numeric policy (no tokenizer)
    elapsedMs: float


def parse_cutoff(value: str) -> int:
    if not ISO_Z.fullmatch(value):
        raise ValueError("cutoff must be YYYY-MM-DDTHH:MM:SSZ")
    t = int(datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    if t % action_task.STEP:
        raise ValueError("cutoff must be a 15-minute UTC boundary")
    return t


def check_position(position: dict, cutoff: int) -> None:
    """Guards stricter than action_task: no state from after the cutoff, consistent flat state."""
    side = position["side"]
    entry, opened, last = position.get("entry_price"), position.get("opened_at"), position.get("last_trade_at")
    if side == "flat":
        if entry is not None or opened is not None:
            raise ValueError("flat position must not carry entry_price or opened_at")
    else:
        if entry is None or not math.isfinite(entry) or entry <= 0 or opened is None:
            raise ValueError("open position requires a positive entry_price and opened_at")
        if last is not None and last < opened:
            raise ValueError("last_trade_at precedes opened_at")
    for t in (opened, last):
        if t is not None and t > cutoff:
            raise ValueError("position timestamps must not be after the cutoff")


def check_candle_open(candles: list[dict]) -> None:
    # BitMEX-derived training opens are the previous close and may lie outside [low, high];
    # action_task validates close/low/high. Only require a positive open here so training-period
    # candles remain replayable through /action for parity checks.
    for c in candles:
        if not c["open"] > 0:
            raise ValueError("invalid OHLCV")


def job_from_request(request: ActionRequest, *, now: float | None = None) -> tuple[dict, int]:
    """Return (job dict from action_task.build_job, cutoff). Raises ValueError on any invalid input."""
    cutoff = parse_cutoff(request.cutoff)
    if cutoff > (time.time() if now is None else now) + FUTURE_SKEW_SECONDS:
        raise ValueError("cutoff is in the future")
    candles = [c.model_dump() for c in request.candles]
    position = request.position.model_dump()
    check_candle_open(candles)
    check_position(position, cutoff)
    job = action_task.build_job(candles=candles, cutoff=cutoff, position=position, market=request.market)
    return job, cutoff


def to_scoring_job(job: dict) -> Job:
    return Job.model_validate(job)


def softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    values = [math.exp(v - top) for v in logits]
    total = sum(values)
    return [v / total for v in values]


def decide(names: list[str], logits: list[float], margin: float) -> str:
    """argmax(logits + b), b = -margin on `hold` only; first maximum wins ties."""
    if len(names) != len(logits) or not logits:
        raise ValueError("logits do not match options")
    if not all(math.isfinite(v) for v in logits):
        raise ValueError("non-finite logits")
    adjusted = [v - margin if n == "hold" else v for n, v in zip(names, logits)]
    best = max(range(len(adjusted)), key=lambda i: adjusted[i])
    return names[best]
