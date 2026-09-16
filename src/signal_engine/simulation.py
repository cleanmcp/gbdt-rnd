"""Bounded contact-now/wait/never simulation over consolidated account state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from statistics import fmean
from typing import Literal

from .contracts import (
    AccountSnapshot,
    Action,
    ActionEvaluation,
    CoverageStatus,
    ModelScore,
    Recommendation,
)


@dataclass(frozen=True)
class ScoreComponents:
    icp_fit: float
    signal_relevance: float
    proxy_event: float | None
    data_confidence: float
    disagreement: bool
    heuristic_only_support: bool

    @property
    def research_priority(self) -> float:
        return self.signal_relevance * (0.5 + 0.5 * self.data_confidence)


def score_components(
    snapshot: AccountSnapshot,
    scores: tuple[ModelScore, ...],
) -> ScoreComponents:
    heuristic = next(
        (score.score for score in scores if score.model_id.startswith("heuristic")),
        0.0,
    )
    learned = [score.score for score in scores if not score.model_id.startswith("heuristic")]
    proxy_event = fmean(learned) if learned else None
    applicable = [
        status for status in snapshot.coverage.values() if status != CoverageStatus.NOT_APPLICABLE
    ]
    observed = sum(status == CoverageStatus.OBSERVED for status in applicable)
    coverage_ratio = observed / len(applicable) if applicable else 0.0
    corroboration = min(
        snapshot.features.get("signals.distinct_types", 0.0) / 3.0,
        1.0,
    )
    data_confidence = 0.6 * coverage_ratio + 0.4 * corroboration
    all_values = [heuristic, *learned]
    disagreement = bool(all_values and max(all_values) - min(all_values) >= 0.35)
    heuristic_only_support = heuristic >= 0.5 and bool(learned) and max(learned) < 0.1
    return ScoreComponents(
        icp_fit=snapshot.features.get("fit.score", 0.0),
        signal_relevance=heuristic,
        proxy_event=proxy_event,
        data_confidence=data_confidence,
        disagreement=disagreement,
        heuristic_only_support=heuristic_only_support,
    )


def recommendation_priority(
    snapshot: AccountSnapshot,
    scores: tuple[ModelScore, ...],
) -> float:
    return score_components(snapshot, scores).research_priority


def evaluate_actions(
    snapshot: AccountSnapshot,
    scores: tuple[ModelScore, ...],
    expected_deal_value: float = 25_000,
    contact_cost: float = 35,
    wait_options: tuple[int, ...] = (7, 14, 30),
) -> tuple[ActionEvaluation, ...]:
    if not scores:
        raise ValueError("simulation requires at least one model score")
    base_probability = fmean(score.score for score in scores)
    sequence_strength = sum(
        value
        for name, value in snapshot.features.items()
        if name.startswith("sequence.") and name.endswith(".weighted")
    )
    signal_strength = snapshot.features.get("signals.weighted_sum", 0.0)
    peak_day = min(14.0, 2.0 + 8.0 * (1 - math.exp(-sequence_strength)))
    half_life = max(7.0, min(90.0, 21.0 + 12.0 * math.log1p(signal_strength)))

    evaluations = []
    for wait_days in (0, *wait_options):
        if peak_day > 0:
            timing_multiplier = math.exp(
                -((wait_days - peak_day) ** 2) / (2 * max(peak_day, 3.0) ** 2)
            )
            timing_multiplier = 0.65 + 0.5 * timing_multiplier
        else:
            timing_multiplier = math.exp(-wait_days / half_life)
        persistence = math.exp(-wait_days / (half_life * 2))
        response_probability = min(base_probability * timing_multiplier * persistence, 1.0)
        expected_value = (
            response_probability * expected_deal_value
            - contact_cost
            - expected_deal_value * 0.0005 * wait_days
        )
        evaluations.append(
            ActionEvaluation(
                action=Action.CONTACT_NOW if wait_days == 0 else Action.WAIT,
                wait_days=wait_days,
                response_probability=response_probability,
                expected_value=expected_value,
            )
        )
    evaluations.append(
        ActionEvaluation(
            action=Action.NEVER,
            wait_days=0,
            response_probability=0,
            expected_value=0,
        )
    )
    return tuple(evaluations)


def build_recommendation(
    rank: int,
    snapshot: AccountSnapshot,
    scores: tuple[ModelScore, ...],
    label_kind: str,
) -> Recommendation:
    components = score_components(snapshot, scores)
    factors = [
        name for score in scores for name, contribution in score.top_factors if contribution > 0
    ][:4]
    if components.heuristic_only_support:
        reason = (
            "Heuristic-only support: Sidekick signal rules are positive, but both "
            "learned proxy models are below 10%. Human review is required."
        )
    elif components.disagreement:
        reason = "Models disagree materially; review the cited signals before any outreach."
    elif factors:
        reason = "Signal relevance supported by " + ", ".join(dict.fromkeys(factors))
    else:
        reason = "Signal relevance is supported by the consolidated trajectory."

    decision_status: Literal[
        "actionable",
        "human_review",
        "insufficient_evidence",
        "research_only",
    ]
    if label_kind == "real_outcome":
        learned_scores = (
            tuple(score for score in scores if not score.model_id.startswith("heuristic")) or scores
        )
        evaluations = evaluate_actions(snapshot, learned_scores)
        best = max(evaluations, key=lambda evaluation: evaluation.expected_value)
        action = best.action
        start = (
            snapshot.as_of.date() + timedelta(days=best.wait_days)
            if best.action != Action.NEVER
            else None
        )
        decision_status = "actionable"
    else:
        evaluations = ()
        action = Action.REVIEW
        start = None
        decision_status = (
            "insufficient_evidence" if components.data_confidence < 0.2 else "human_review"
        )
    return Recommendation(
        account_id=snapshot.account_id,
        rank=rank,
        action=action,
        best_window_start=start,
        best_window_end=start + timedelta(days=3) if start else None,
        opportunity_score=components.signal_relevance,
        icp_fit_score=components.icp_fit,
        signal_relevance_score=components.signal_relevance,
        proxy_event_score=components.proxy_event,
        data_confidence_score=components.data_confidence,
        model_disagreement=components.disagreement,
        decision_status=decision_status,
        model_scores=scores,
        action_evaluations=evaluations,
        reason=reason,
        evidence_fact_ids=snapshot.evidence_fact_ids[:20],
    )
