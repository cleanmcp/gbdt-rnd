from datetime import UTC, datetime
from pathlib import Path

from signal_engine.config import load_decision_context
from signal_engine.contracts import TrainingRow
from signal_engine.portfolio import propose_portfolio_weights

ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_smooths_small_samples_and_preserves_floor() -> None:
    context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    rows = [
        TrainingRow(
            account_id=f"account-{index}",
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            features={
                "signal.sba_loan.active_count": float(index < 4),
                "signal.osha_injury_summary.active_count": float(index < 8),
            },
            label=int(index in {0, 1, 5}),
            label_kind="synthetic_oracle",
        )
        for index in range(20)
    ]
    weights = propose_portfolio_weights(rows, context.signal_policy)
    assert weights
    assert all(0.05 <= weight.proposed_weight <= 1 for weight in weights)
    sba = next(weight for weight in weights if weight.signal_type == "sba_loan")
    assert sba.observations == 4
    assert sba.proposed_weight != 1
