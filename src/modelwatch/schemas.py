"""Strict event contracts; raw request text is deliberately excluded."""
from datetime import datetime, timezone
from typing import Annotated, Literal
import json
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, TypeAdapter, field_validator, model_validator

Identifier = Annotated[str, Field(strict=True, min_length=1, max_length=160, pattern=r'^[A-Za-z0-9_.:-]+$')]
Label = Annotated[str, Field(strict=True, min_length=1, max_length=160)]

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Model(Contract):
    model_id: Identifier
    name: Annotated[str, Field(strict=True, min_length=1, max_length=120)]
    version: Identifier
    classes: list[Label] = Field(min_length=2, max_length=1000)
    metadata: dict = Field(default_factory=dict)

    @field_validator('classes')
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError('Class labels must be unique')
        return value

    @field_validator('metadata')
    @classmethod
    def bounded_metadata(cls, value):
        if len(json.dumps(value, allow_nan=False)) > 32000:
            raise ValueError('Metadata exceeds 32 KB')
        return value

class Run(Contract):
    run_id: Identifier
    model_id: Identifier
    role: Literal['reference', 'monitor']
    source: Literal['benchmark_replay', 'live']
    metadata: dict = Field(default_factory=dict)

    @field_validator('metadata')
    @classmethod
    def bounded_metadata(cls, value):
        value = Model.bounded_metadata(value)
        for key in ('reference_run_id', 'replay_id'):
            if value.get(key) is not None:
                TypeAdapter(Identifier).validate_python(value[key])
        return value

class Features(Contract):
    text_length: int = Field(strict=True, ge=0, le=1000000)
    word_count: int = Field(strict=True, ge=0, le=1000000)

class Event(Contract):
    event_id: Identifier
    run_id: Identifier
    occurred_at: AwareDatetime
    status: Literal['ok', 'error']
    latency_ms: float = Field(ge=0, le=3600000, strict=True)
    predicted_label: Label | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, strict=True)
    features: Features
    sample_id: Identifier | None = None
    error_type: Identifier | None = None

    @field_validator('occurred_at')
    @classmethod
    def utc(cls, value):
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def coherent(self):
        if self.status == 'ok' and (self.predicted_label is None or self.confidence is None or self.error_type is not None):
            raise ValueError('Successful events require prediction and confidence, without error_type')
        if self.status == 'error' and (self.predicted_label is not None or self.confidence is not None or self.error_type is None):
            raise ValueError('Error events require error_type and no prediction/confidence')
        return self

class Feedback(Contract):
    event_id: Identifier
    label: Label
    source: Literal['official_dataset', 'human_review']
    labeled_at: AwareDatetime

    @field_validator('labeled_at')
    @classmethod
    def utc(cls, value):
        return value.astimezone(timezone.utc)
