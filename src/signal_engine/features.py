"""Point-in-time account consolidation conditioned on an ICP and product goal."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from statistics import fmean

from .contracts import (
    AccountRecord,
    AccountSnapshot,
    CoverageStatus,
    FactSheet,
    GoalSpec,
    IcpSpec,
    NormalizedEvent,
    SequenceRule,
    SignalPolicy,
)
from .hashing import sha256_json


def account_fits_icp(account: AccountRecord, icp: IcpSpec) -> tuple[bool, float]:
    checks: list[float] = []
    hard_checks: list[bool] = []
    if icp.naics_prefixes:
        naics_match = bool(
            account.naics_code
            and any(account.naics_code.startswith(prefix) for prefix in icp.naics_prefixes)
        )
        checks.append(float(naics_match))
        hard_checks.append(naics_match)
    if icp.geographies:
        geography_match = bool(account.state and account.state in icp.geographies)
        checks.append(float(geography_match))
        hard_checks.append(geography_match)
    if icp.employee_min is not None:
        checks.append(
            0.5
            if account.employee_count is None
            else float(account.employee_count >= icp.employee_min)
        )
    if icp.employee_max is not None:
        checks.append(
            0.5
            if account.employee_count is None
            else float(account.employee_count <= icp.employee_max)
        )
    score = fmean(checks) if checks else 1.0
    hard_fit = all(hard_checks) if hard_checks else True
    return hard_fit, score


def _numeric_facts(sheet: FactSheet) -> dict[str, float]:
    values: dict[str, float] = {}
    for fact in sheet.facts:
        if isinstance(fact.value, (bool, float, int)):
            values[fact.name] = float(fact.value)
    return values


def _latest_time(sheet: FactSheet) -> datetime:
    times = [
        evidence.observed_at
        for fact in sheet.facts
        for evidence in fact.evidence_refs
        if evidence.observed_at is not None
    ]
    return max(times, default=sheet.as_of)


def _sequence_matches(
    rule: SequenceRule,
    events: list[NormalizedEvent],
    as_of: datetime,
) -> tuple[int, float]:
    visible = sorted(
        (event for event in events if event.available_at <= as_of),
        key=lambda event: (event.occurred_at, event.event_id),
    )
    matches = 0
    most_recent_end: datetime | None = None
    start_index = 0
    while start_index < len(visible):
        first_type = rule.ordered_signal_types[0]
        try:
            first_index = next(
                index
                for index in range(start_index, len(visible))
                if visible[index].signal_type == first_type
            )
        except StopIteration:
            break
        previous = visible[first_index]
        cursor = first_index + 1
        complete = True
        for signal_type in rule.ordered_signal_types[1:]:
            found: NormalizedEvent | None = None
            while cursor < len(visible):
                candidate = visible[cursor]
                cursor += 1
                gap = (candidate.occurred_at - previous.occurred_at).days
                if gap > rule.max_gap_days:
                    break
                if candidate.signal_type == signal_type:
                    found = candidate
                    break
            if found is None:
                complete = False
                break
            previous = found
        if complete:
            matches += 1
            most_recent_end = previous.occurred_at
            start_index = cursor
        else:
            start_index = first_index + 1
    recency = (
        0.0
        if most_recent_end is None
        else math.exp(-max((as_of - most_recent_end).days, 0) / rule.max_gap_days)
    )
    return matches, recency


def build_account_snapshot(
    account: AccountRecord,
    factsheets: list[FactSheet],
    events: list[NormalizedEvent],
    icp: IcpSpec,
    goal: GoalSpec,
    policy: SignalPolicy,
    as_of: datetime,
    collected_signal_types: set[str] | None = None,
    include_story_features: bool = False,
) -> AccountSnapshot:
    hard_fit, fit_score = account_fits_icp(account, icp)
    features: dict[str, float] = {
        "fit.hard_pass": float(hard_fit),
        "fit.score": fit_score,
        "account.employee_count": float(account.employee_count or 0),
    }
    coverage: dict[str, CoverageStatus] = {}
    by_type: dict[str, list[FactSheet]] = defaultdict(list)
    for sheet in factsheets:
        if sheet.account_id == account.account_id and sheet.as_of <= as_of:
            by_type[sheet.signal_type].append(sheet)

    active_signal_count = 0
    weighted_signal_sum = 0.0
    evidence_ids: set[str] = set()
    # Deterministic StoryCard phases (see stories.deterministic_signal_story); only emitted as
    # features when explicitly requested so ablations can measure them before promotion.
    story_totals: dict[str, int] = {"active": 0, "escalating": 0, "resolved": 0}
    for entry in policy.entries:
        applicable = not entry.applicability_naics_prefixes or bool(
            account.naics_code
            and any(
                account.naics_code.startswith(prefix)
                for prefix in entry.applicability_naics_prefixes
            )
        )
        sheets = by_type.get(entry.signal_type, [])
        if not applicable:
            coverage[entry.signal_type] = CoverageStatus.NOT_APPLICABLE
            continue
        if collected_signal_types is not None and entry.signal_type not in collected_signal_types:
            coverage[entry.signal_type] = CoverageStatus.NOT_COLLECTED
            continue
        if not sheets:
            coverage[entry.signal_type] = CoverageStatus.NOT_OBSERVED
            continue
        coverage[entry.signal_type] = CoverageStatus.OBSERVED
        active_signal_count += len(sheets)
        instance_scores: list[float] = []
        fact_values: dict[str, list[float]] = defaultdict(list)
        phase_counts: dict[str, int] = {"active": 0, "escalating": 0, "resolved": 0}
        for sheet in sheets:
            values = _numeric_facts(sheet)
            age_days = max((as_of - _latest_time(sheet)).total_seconds() / 86_400, 0)
            terminal = values.get("is_terminal", 0.0)
            material_changes = values.get("material_change_count", 0.0)
            phase_counts[
                "resolved" if terminal else "escalating" if material_changes >= 2 else "active"
            ] += 1
            stage_multiplier = (
                0.05
                if terminal
                else min(
                    1.75,
                    1.0 + 0.08 * material_changes,
                )
            )
            decayed = (
                values.get("latest_strength", 0.0)
                * stage_multiplier
                * math.exp(-age_days / entry.half_life_days)
            )
            instance_scores.append(decayed)
            for name, value in values.items():
                fact_values[name].append(value)
            evidence_ids.update(fact.fact_id for fact in sheet.facts)
        signal_score = entry.initial_weight * sum(instance_scores)
        weighted_signal_sum += signal_score
        prefix = f"signal.{entry.signal_type}"
        features[f"{prefix}.active_count"] = float(len(sheets))
        features[f"{prefix}.decayed_score"] = signal_score
        features[f"{prefix}.max_instance_score"] = max(instance_scores)
        for name, fact_numbers in fact_values.items():
            features[f"{prefix}.{name}.max"] = max(fact_numbers)
            features[f"{prefix}.{name}.mean"] = fmean(fact_numbers)
        for phase, count in phase_counts.items():
            story_totals[phase] += count
            if include_story_features:
                features[f"story.{entry.signal_type}.phase_{phase}"] = float(count)

    if include_story_features:
        total_stories = sum(story_totals.values())
        for phase, count in story_totals.items():
            features[f"story.phase_{phase}.share"] = count / total_stories if total_stories else 0.0
        features["story.phase_escalating.any"] = float(story_totals["escalating"] > 0)

    features["signals.active_count"] = float(active_signal_count)
    features["signals.distinct_types"] = float(
        sum(status == CoverageStatus.OBSERVED for status in coverage.values())
    )
    features["signals.weighted_sum"] = weighted_signal_sum
    features["signals.corroboration"] = min(
        features["signals.distinct_types"] / 4.0,
        1.0,
    )
    account_events = [event for event in events if event.account_id == account.account_id]
    for rule in policy.sequence_rules:
        count, recency = _sequence_matches(rule, account_events, as_of)
        features[f"sequence.{rule.rule_id}.count"] = float(count)
        features[f"sequence.{rule.rule_id}.recency"] = recency
        features[f"sequence.{rule.rule_id}.weighted"] = count * recency * rule.weight

    snapshot_payload = {
        "account_id": account.account_id,
        "as_of": as_of,
        "icp_version": icp.icp_version,
        "goal_version": goal.goal_version,
        "features": features,
        "coverage": coverage,
        "evidence_fact_ids": sorted(evidence_ids),
    }
    return AccountSnapshot(
        account_id=account.account_id,
        as_of=as_of,
        icp_version=icp.icp_version,
        goal_version=goal.goal_version,
        features=features,
        coverage=coverage,
        evidence_fact_ids=tuple(sorted(evidence_ids)),
        snapshot_hash=sha256_json(snapshot_payload),
    )
