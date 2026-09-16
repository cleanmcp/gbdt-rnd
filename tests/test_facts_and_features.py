from datetime import UTC, datetime, timedelta
from pathlib import Path

from signal_engine.config import load_decision_context
from signal_engine.facts import build_factsheet
from signal_engine.features import build_account_snapshot
from signal_engine.fixtures import build_fixture_corpus
from signal_engine.signal_registry import SignalRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_factsheet_semantic_hash_ignores_clock_only_age() -> None:
    registry = SignalRegistry.from_directory(ROOT / "signals")
    fixture = build_fixture_corpus(size=20)
    instance_events = [event for event in fixture.events if event.signal_type == "sba_loan"][:2]
    assert instance_events
    definition = registry.get("sba_loan")
    first = build_factsheet(
        definition,
        instance_events,
        datetime(2026, 6, 1, tzinfo=UTC),
    )
    later = build_factsheet(
        definition,
        instance_events,
        datetime(2026, 6, 2, tzinfo=UTC),
    )
    assert first.semantic_hash == later.semantic_hash
    first_age = next(fact.value for fact in first.facts if fact.name == "age_days")
    later_age = next(fact.value for fact in later.facts if fact.name == "age_days")
    assert later_age == first_age + 1


def test_same_world_facts_produce_different_icp_fit() -> None:
    registry = SignalRegistry.from_directory(ROOT / "signals")
    context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    fixture = build_fixture_corpus(size=1)
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    by_instance: dict[str, list] = {}
    for event in fixture.events:
        by_instance.setdefault(event.signal_instance_id, []).append(event)
    sheets = [
        build_factsheet(registry.get(events[0].signal_type), events, as_of)
        for events in by_instance.values()
        if min(event.available_at for event in events) <= as_of
    ]
    manufacturing = build_account_snapshot(
        fixture.accounts[0],
        sheets,
        fixture.events,
        context.icp,
        context.goal,
        context.signal_policy,
        as_of,
    )
    software_icp = context.icp.model_copy(
        update={"icp_version": "software-v1", "naics_prefixes": ("51",)}
    )
    software = build_account_snapshot(
        fixture.accounts[0],
        sheets,
        fixture.events,
        software_icp,
        context.goal,
        context.signal_policy,
        as_of,
    )
    assert manufacturing.features["fit.hard_pass"] == 1
    assert software.features["fit.hard_pass"] == 0


def test_future_events_never_enter_point_in_time_factsheet() -> None:
    registry = SignalRegistry.from_directory(ROOT / "signals")
    fixture = build_fixture_corpus(size=40)
    event = next(
        candidate for candidate in fixture.events if candidate.signal_type == "job_posting"
    )
    visible_at = event.available_at - timedelta(seconds=1)
    definition = registry.get(event.signal_type)
    try:
        build_factsheet(definition, [event], visible_at)
    except ValueError as error:
        assert "point-in-time-visible" in str(error)
    else:
        raise AssertionError("future event leaked into the FactSheet")
