from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .settings import MAX_JOBS, MAX_OPTIONS

StructuredValue = str | dict[str, JsonValue] | list[JsonValue]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Option(StrictModel):
    name: Annotated[str, Field(min_length=1, max_length=256)]
    description: StructuredValue | None = None


class Job(StrictModel):
    state: StructuredValue
    instructions: StructuredValue
    options: Annotated[list[Option], Field(min_length=1, max_length=MAX_OPTIONS)]

    @model_validator(mode="after")
    def distinct_option_names(self) -> "Job":
        if len({option.name for option in self.options}) != len(self.options):
            raise ValueError("option names must be unique within a job")
        return self


class ScoreRequest(StrictModel):
    jobs: Annotated[list[Job], Field(min_length=1, max_length=MAX_JOBS)]


class JobScore(StrictModel):
    logits: list[float]
    inputTokens: int


class ScoreResponse(StrictModel):
    model: str
    revision: str
    scores: list[JobScore]
    elapsedMs: float
