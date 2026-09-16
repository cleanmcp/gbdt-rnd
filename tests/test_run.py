import asyncio
from datetime import UTC, datetime
from pathlib import Path

from signal_engine.config import load_decision_context
from signal_engine.contracts import RunState
from signal_engine.fixtures import build_fixture_corpus
from signal_engine.runs import RunCoordinator, build_experiment_spec
from signal_engine.signal_registry import SignalRegistry
from signal_engine.store import ExperimentStore

ROOT = Path(__file__).resolve().parents[1]


def test_complete_experiment_is_durable_and_idempotent(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "test.duckdb")
    registry = SignalRegistry.from_directory(ROOT / "signals")
    fixture = build_fixture_corpus(size=70)
    coordinator = RunCoordinator(store=store, registry=registry, fixture=fixture)
    for account in fixture.accounts:
        store.upsert_account(account)
    for event in fixture.events:
        store.append_event(event)
    context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    spec = build_experiment_spec(
        context,
        as_of=datetime(2026, 6, 30, tzinfo=UTC),
        capacity=8,
        model_ids=("heuristic-v1", "lightgbm-v1"),
        max_llm_story_cards=8,
    )
    run, created = coordinator.request(spec)
    assert created
    asyncio.run(coordinator.execute(run.run_id))

    completed = store.run(run.run_id)
    assert completed is not None
    assert completed.state == "completed", [
        event.payload for event in store.run_events(run.run_id) if event.kind == "run.failed"
    ]
    events = store.run_events(run.run_id)
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert events[-1].kind == "run.completed"
    assert store.recommendations(run.run_id)
    assert store.benchmarks(run.run_id)

    replayed, replay_created = coordinator.request(
        spec.model_copy(update={"experiment_id": "different-request-id"})
    )
    assert not replay_created
    assert replayed.run_id == run.run_id
    store.close()


def test_reconcile_marks_interrupted_run_indeterminate(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "recovery.duckdb")
    registry = SignalRegistry.from_directory(ROOT / "signals")
    coordinator = RunCoordinator(store=store, registry=registry)
    context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    spec = build_experiment_spec(
        context,
        as_of=datetime(2026, 6, 30, tzinfo=UTC),
        model_ids=("heuristic-v1",),
    )
    run, _ = coordinator.request(spec)
    store.save_run(
        run.model_copy(
            update={
                "state": RunState.RUNNING,
                "started_at": datetime(2026, 6, 30, tzinfo=UTC),
            }
        )
    )
    assert coordinator.reconcile_interrupted_runs() == 1
    recovered = store.run(run.run_id)
    assert recovered is not None
    assert recovered.state == "indeterminate"
    assert recovered.error_code == "PROCESS_RESTARTED"
    assert store.run_events(run.run_id)[-1].kind == "run.indeterminate"
    store.close()
