import asyncio
from datetime import UTC, datetime
from pathlib import Path

from signal_engine.config import load_decision_context
from signal_engine.contracts import AccountSnapshot, CoverageStatus
from signal_engine.store import ExperimentStore
from signal_engine.stories import StoryBudget, TemplateNarrator, account_story

ROOT = Path(__file__).resolve().parents[1]


def test_account_story_is_cached_without_spending_second_budget_claim(
    tmp_path: Path,
) -> None:
    context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    store = ExperimentStore(tmp_path / "stories.duckdb")
    snapshot = AccountSnapshot(
        account_id="acme",
        as_of=datetime(2026, 6, 30, tzinfo=UTC),
        icp_version=context.icp.icp_version,
        goal_version=context.goal.goal_version,
        features={"signals.weighted_sum": 1.2},
        coverage={"sba_loan": CoverageStatus.OBSERVED},
        evidence_fact_ids=("loan:1:amount",),
        snapshot_hash="abc123",
    )
    budget = StoryBudget(1)

    async def generate_twice():
        first = await account_story(
            store=store,
            provider=TemplateNarrator(),
            budget=budget,
            product=context.product,
            icp=context.icp,
            snapshot=snapshot,
            fact_payload=[],
        )
        second = await account_story(
            store=store,
            provider=TemplateNarrator(),
            budget=budget,
            product=context.product,
            icp=context.icp,
            snapshot=snapshot,
            fact_payload=[],
        )
        return first, second

    first, second = asyncio.run(generate_twice())
    assert first is not None
    assert first == second
    assert budget.used == 1
    store.close()
