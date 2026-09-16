from datetime import UTC, datetime

from signal_engine.contracts import (
    AccountSnapshot,
    Action,
    CoverageStatus,
    ModelScore,
)
from signal_engine.simulation import build_recommendation


def _snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        account_id="manufacturer-1",
        as_of=datetime(2026, 6, 30, tzinfo=UTC),
        icp_version="sidekick-v1",
        goal_version="goal-v1",
        features={
            "fit.score": 1.0,
            "signals.distinct_types": 2.0,
            "signals.weighted_sum": 1.2,
        },
        coverage={
            "sba_loan": CoverageStatus.OBSERVED,
            "osha_injury_summary": CoverageStatus.OBSERVED,
            "epa_tri_report": CoverageStatus.NOT_COLLECTED,
            "job_posting": CoverageStatus.NOT_COLLECTED,
            "usaspending_award": CoverageStatus.NOT_COLLECTED,
        },
        evidence_fact_ids=("fact-1",),
        snapshot_hash="snapshot-hash",
    )


def test_public_proxy_disagreement_requires_review_without_ev_claims() -> None:
    scores = (
        ModelScore(model_id="heuristic-v1", account_id="manufacturer-1", score=0.6),
        ModelScore(model_id="lightgbm-v1", account_id="manufacturer-1", score=0.0),
        ModelScore(model_id="tabpfn-local-v2", account_id="manufacturer-1", score=0.01),
    )
    recommendation = build_recommendation(
        1,
        _snapshot(),
        scores,
        "public_future_event",
    )
    assert recommendation.action == Action.REVIEW
    assert recommendation.decision_status == "human_review"
    assert recommendation.action_evaluations == ()
    assert recommendation.signal_relevance_score == 0.6
    assert recommendation.proxy_event_score == 0.005
    assert recommendation.model_disagreement
    assert "Heuristic-only support" in recommendation.reason


def test_real_outcome_scores_can_drive_action_simulation() -> None:
    scores = (
        ModelScore(model_id="lightgbm-v1", account_id="manufacturer-1", score=0.4),
        ModelScore(model_id="tabpfn-local-v2", account_id="manufacturer-1", score=0.5),
    )
    recommendation = build_recommendation(1, _snapshot(), scores, "real_outcome")
    assert recommendation.decision_status == "actionable"
    assert recommendation.action != Action.REVIEW
    assert recommendation.action_evaluations
