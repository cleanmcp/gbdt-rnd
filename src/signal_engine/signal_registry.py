"""Declarative signal registry: new signal types are data, not engine branches."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import SignalShape
from .hashing import sha256_json


class NumericFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    unit: str | None = None
    reducers: tuple[Literal["latest", "max", "mean", "sum", "delta", "slope"], ...] = ("latest",)


class SignalDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    signal_type: str
    version: int = Field(ge=1)
    description: str
    shape: SignalShape
    source_ids: tuple[str, ...]
    durable: bool
    identity_fields: tuple[str, ...]
    state_field: str | None = None
    terminal_states: tuple[str, ...] = ()
    material_kinds: tuple[str, ...]
    numeric_features: tuple[NumericFeature, ...] = ()
    default_half_life_days: float = Field(gt=0)
    applicability_naics_prefixes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_numeric_features(self) -> Self:
        fields = [feature.field for feature in self.numeric_features]
        if len(fields) != len(set(fields)):
            raise ValueError("numeric feature fields must be unique")
        return self

    @property
    def registry_id(self) -> str:
        return f"{self.signal_type}@{self.version}"

    @property
    def contract_hash(self) -> str:
        return sha256_json(self)


class SignalRegistry:
    def __init__(self, definitions: tuple[SignalDefinition, ...]):
        by_type = {definition.signal_type: definition for definition in definitions}
        if len(by_type) != len(definitions):
            raise ValueError("duplicate signal_type in registry")
        self._definitions = by_type

    @classmethod
    def from_directory(cls, path: Path) -> SignalRegistry:
        definitions = []
        for file_path in sorted(path.glob("*.yaml")):
            raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            definitions.append(SignalDefinition.model_validate(raw))
        if not definitions:
            raise ValueError(f"signal registry is empty: {path}")
        return cls(tuple(definitions))

    def get(self, signal_type: str) -> SignalDefinition:
        try:
            return self._definitions[signal_type]
        except KeyError as error:
            raise KeyError(f"unknown signal type: {signal_type}") from error

    def all(self) -> tuple[SignalDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    def manifest(self) -> dict[str, str]:
        return {definition.registry_id: definition.contract_hash for definition in self.all()}
