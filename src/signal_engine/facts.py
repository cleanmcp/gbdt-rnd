"""Deterministic per-instance FactSheets for all four signal shapes."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from itertools import pairwise
from statistics import fmean
from typing import Any

import numpy as np

from .contracts import (
    CoverageStatus,
    EvidenceRef,
    Fact,
    FactSheet,
    NormalizedEvent,
)
from .hashing import sha256_json
from .signal_registry import NumericFeature, SignalDefinition


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)) and np.isfinite(float(value)):
        return float(value)
    return None


def _evidence(event: NormalizedEvent, field: str | None = None) -> EvidenceRef:
    return EvidenceRef(
        source_id=event.source_id,
        source_record_id=event.source_record_id,
        field=field,
        observed_at=event.available_at,
    )


def _fact(
    instance_id: str,
    name: str,
    value: bool | float | str | None,
    evidence: tuple[EvidenceRef, ...],
    unit: str | None = None,
) -> Fact:
    return Fact(
        fact_id=f"{instance_id}:{name}",
        name=name,
        value=value,
        unit=unit,
        evidence_refs=evidence,
    )


def _numeric_facts(
    instance_id: str,
    feature: NumericFeature,
    events: list[NormalizedEvent],
) -> list[Fact]:
    observations = [
        (event, number)
        for event in events
        if (number := _number(event.payload.get(feature.field))) is not None
    ]
    if not observations:
        return []
    values = [number for _, number in observations]
    evidence = tuple(_evidence(event, feature.field) for event, _ in observations)
    result: list[Fact] = []
    for reducer in feature.reducers:
        value: float
        if reducer == "latest":
            value = values[-1]
        elif reducer == "max":
            value = max(values)
        elif reducer == "mean":
            value = fmean(values)
        elif reducer == "sum":
            value = sum(values)
        elif reducer == "delta":
            value = values[-1] - values[0]
        elif reducer == "slope":
            if len(values) < 2:
                value = 0.0
            else:
                origin = observations[0][0].occurred_at
                days = [
                    max((event.occurred_at - origin).total_seconds() / 86_400, 0.0)
                    for event, _ in observations
                ]
                value = float(np.polyfit(days, values, 1)[0]) if len(set(days)) > 1 else 0.0
        else:  # pragma: no cover - Pydantic closes the reducer vocabulary.
            raise ValueError(f"unsupported reducer: {reducer}")
        result.append(
            _fact(
                instance_id,
                f"{feature.field}.{reducer}",
                round(value, 6),
                evidence,
                feature.unit,
            )
        )
    return result


def build_factsheet(
    definition: SignalDefinition,
    raw_events: Iterable[NormalizedEvent],
    as_of: datetime,
) -> FactSheet:
    events = sorted(
        (event for event in raw_events if event.available_at <= as_of),
        key=lambda event: (event.occurred_at, event.available_at, event.event_id),
    )
    if not events:
        raise ValueError("a FactSheet requires at least one point-in-time-visible event")
    instance_ids = {event.signal_instance_id for event in events}
    signal_types = {event.signal_type for event in events}
    if len(instance_ids) != 1 or signal_types != {definition.signal_type}:
        raise ValueError("FactSheet events must belong to one registered signal instance")

    first, latest = events[0], events[-1]
    instance_id = first.signal_instance_id
    identity = {
        field: latest.payload.get(field)
        for field in definition.identity_fields
        if isinstance(latest.payload.get(field), (str, int, float))
        or latest.payload.get(field) is None
    }
    current_state: dict[str, str | int | float | bool | None] = {
        "kind": latest.kind,
        "strength": latest.strength,
    }
    if definition.state_field:
        state = latest.payload.get(definition.state_field)
        if isinstance(state, (str, int, float, bool)) or state is None:
            current_state["state"] = state

    all_evidence = tuple(_evidence(event) for event in events)
    facts = [
        _fact(instance_id, "event_count", len(events), all_evidence, "events"),
        _fact(
            instance_id,
            "age_days",
            round((as_of - first.occurred_at).total_seconds() / 86_400, 3),
            (_evidence(first),),
            "days",
        ),
        _fact(
            instance_id,
            "days_since_last_change",
            round((as_of - latest.occurred_at).total_seconds() / 86_400, 3),
            (_evidence(latest),),
            "days",
        ),
        _fact(instance_id, "latest_strength", latest.strength, (_evidence(latest),)),
        _fact(
            instance_id,
            "maximum_strength",
            max(event.strength for event in events),
            all_evidence,
        ),
        _fact(
            instance_id,
            "source_latency_days",
            round((latest.delivered_at - latest.occurred_at).total_seconds() / 86_400, 3),
            (_evidence(latest),),
            "days",
        ),
    ]
    material_events = [event for event in events if event.kind in definition.material_kinds]
    facts.append(
        _fact(
            instance_id,
            "material_change_count",
            len(material_events),
            tuple(_evidence(event) for event in material_events),
            "events",
        )
    )
    if definition.state_field:
        states = [str(event.payload.get(definition.state_field, "unknown")) for event in events]
        transitions = sum(left != right for left, right in pairwise(states))
        facts.extend(
            [
                _fact(instance_id, "current_state", states[-1], (_evidence(latest),)),
                _fact(instance_id, "state_transition_count", transitions, all_evidence),
                _fact(
                    instance_id,
                    "is_terminal",
                    states[-1] in definition.terminal_states,
                    (_evidence(latest),),
                ),
            ]
        )
    for feature in definition.numeric_features:
        facts.extend(_numeric_facts(instance_id, feature, events))

    semantic_payload = {
        "signal_type": definition.signal_type,
        "identity": identity,
        "current_state": current_state,
        "material_events": [
            {
                "kind": event.kind,
                "occurred_at": event.occurred_at,
                "payload": event.payload,
            }
            for event in material_events
        ],
        "facts": [
            fact.model_dump(mode="json")
            for fact in facts
            if fact.name not in {"age_days", "days_since_last_change"}
        ],
    }
    semantic_hash = sha256_json(semantic_payload)
    return FactSheet(
        factsheet_id=f"{instance_id}:{as_of.isoformat()}",
        signal_instance_id=instance_id,
        signal_type=definition.signal_type,
        account_id=first.account_id,
        shape=definition.shape,
        as_of=as_of,
        coverage=CoverageStatus.OBSERVED,
        identity=identity,
        current_state=current_state,
        facts=tuple(facts),
        semantic_hash=semantic_hash,
    )
