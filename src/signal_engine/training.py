"""Point-in-time materialization and dual non-production R&D labels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import DecisionContext
from .contracts import (
    AccountRecord,
    AccountSnapshot,
    FactSheet,
    NormalizedEvent,
    TrainingRow,
)
from .facts import build_factsheet
from .features import build_account_snapshot
from .signal_registry import SignalRegistry


@dataclass(frozen=True)
class TrainingMaterialization:
    snapshots: list[AccountSnapshot]
    rows: list[TrainingRow]


def materialize_at(
    *,
    accounts: list[AccountRecord],
    events: list[NormalizedEvent],
    as_of: datetime,
    context: DecisionContext,
    registry: SignalRegistry,
    include_story_features: bool = False,
) -> list[AccountSnapshot]:
    by_instance: dict[str, list[NormalizedEvent]] = defaultdict(list)
    for event in events:
        if event.available_at <= as_of:
            by_instance[event.signal_instance_id].append(event)
    factsheets = [
        build_factsheet(registry.get(instance_events[0].signal_type), instance_events, as_of)
        for instance_events in by_instance.values()
    ]
    collected = {event.signal_type for event in events}
    sheets_by_account: dict[str, list[FactSheet]] = defaultdict(list)
    events_by_account: dict[str, list[NormalizedEvent]] = defaultdict(list)
    for sheet in factsheets:
        sheets_by_account[sheet.account_id].append(sheet)
    for event in events:
        events_by_account[event.account_id].append(event)
    return [
        build_account_snapshot(
            account,
            sheets_by_account[account.account_id],
            events_by_account[account.account_id],
            context.icp,
            context.goal,
            context.signal_policy,
            as_of,
            collected,
            include_story_features=include_story_features,
        )
        for account in accounts
    ]


def build_training_materialization(
    *,
    accounts: list[AccountRecord],
    events: list[NormalizedEvent],
    dates: tuple[datetime, ...],
    context: DecisionContext,
    registry: SignalRegistry,
    latent_opportunity: dict[str, float] | None = None,
    label_kind: str = "synthetic_oracle",
    include_story_features: bool = False,
) -> TrainingMaterialization:
    all_snapshots: list[AccountSnapshot] = []
    rows: list[TrainingRow] = []
    events_by_account: dict[str, list[NormalizedEvent]] = defaultdict(list)
    for event in events:
        events_by_account[event.account_id].append(event)
    for as_of in dates:
        snapshots = materialize_at(
            accounts=accounts,
            events=events,
            as_of=as_of,
            context=context,
            registry=registry,
            include_story_features=include_story_features,
        )
        all_snapshots.extend(snapshots)
        horizon = as_of + timedelta(days=context.goal.horizon_days)
        for snapshot in snapshots:
            if label_kind == "synthetic_oracle":
                if latent_opportunity is None:
                    raise ValueError("synthetic oracle labels require latent opportunity values")
                sequence_signal = sum(
                    value
                    for name, value in snapshot.features.items()
                    if name.startswith("sequence.") and name.endswith(".weighted")
                )
                label = int(
                    latent_opportunity.get(snapshot.account_id, 0) >= 0.58
                    and (sequence_signal > 0 or snapshot.features["signals.distinct_types"] >= 2)
                )
            elif label_kind == "public_future_event":
                label = int(
                    any(
                        as_of < event.occurred_at <= horizon
                        and event.signal_type in {"sba_loan", "usaspending_award", "job_posting"}
                        for event in events_by_account[snapshot.account_id]
                    )
                )
            else:
                raise ValueError(f"unsupported R&D label kind: {label_kind}")
            rows.append(
                TrainingRow(
                    account_id=snapshot.account_id,
                    as_of=as_of,
                    features=snapshot.features,
                    label=label,
                    label_kind=label_kind,  # type: ignore[arg-type]
                )
            )
    return TrainingMaterialization(all_snapshots, rows)
