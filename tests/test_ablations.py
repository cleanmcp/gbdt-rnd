from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from signal_engine.ablations import (
    ARCHITECTURE_ARMS,
    ARCHITECTURE_DATES,
    ARCHITECTURE_FAMILIES,
    MODEL_SPECS,
    PROTOCOLS,
    MaskSpec,
    PositiveRateScorer,
    aggregate_cells,
    apply_feature_mask,
    apply_post_split_control,
    evaluate_cell,
    population_filter,
    pristine_labels,
    shuffle_history_timestamps,
)
from signal_engine.config import load_decision_context
from signal_engine.contracts import AccountSnapshot, TrainingRow
from signal_engine.fixtures import build_fixture_corpus
from signal_engine.hashing import sha256_json
from signal_engine.outcome_benchmarks import LAPTOP_PROFILE
from signal_engine.signal_registry import SignalRegistry
from signal_engine.training import materialize_at

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_rows(n: int = 600, seed: int = 3) -> tuple[list[TrainingRow], dict]:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    snapshots = {}
    for index in range(n):
        signal = rng.normal()
        noise = rng.normal()
        features = {
            "fit.score": 1.0,
            "contact.euribor3m": float(signal),
            "contact.age": float(noise),
            "contact.poutcome.success": float(rng.random() < 0.2),
        }
        label = int(signal + 0.5 * features["contact.poutcome.success"] + 0.3 * noise > 0.8)
        as_of = start + timedelta(days=index)
        account_id = f"acct-{index:04d}"
        rows.append(
            TrainingRow(
                account_id=account_id,
                as_of=as_of,
                features=features,
                label=label,
                label_kind="real_outcome",
            )
        )
        snapshots[(account_id, as_of.isoformat())] = AccountSnapshot(
            account_id=account_id,
            as_of=as_of,
            icp_version="test",
            goal_version="test",
            features=features,
            coverage={},
            evidence_fact_ids=(),
            snapshot_hash=sha256_json({"a": account_id}),
        )
    return rows, snapshots


def test_mask_spec_parses_and_filters_families() -> None:
    families = {"macro": lambda n: "euribor" in n, "prior": lambda n: "poutcome" in n}
    assert MaskSpec.parse("none").keep(families)("contact.euribor3m")
    drop = MaskSpec.parse("drop:macro").keep(families)
    assert not drop("contact.euribor3m") and drop("contact.age")
    only = MaskSpec.parse("only:macro+prior").keep(families)
    assert only("contact.euribor3m") and only("contact.poutcome.success")
    assert not only("contact.age") and only("fit.score")
    with pytest.raises(ValueError):
        MaskSpec.parse("keep:macro")
    with pytest.raises(ValueError):
        MaskSpec.parse("drop:unknown").keep(families)


def test_apply_feature_mask_drops_columns_from_rows_and_snapshots() -> None:
    rows, snapshots = _synthetic_rows(20)
    masked_rows, masked_snapshots, kept = apply_feature_mask(
        rows, snapshots, lambda name: "euribor" not in name
    )
    assert kept == 3
    assert all("contact.euribor3m" not in row.features for row in masked_rows)
    assert all("contact.euribor3m" not in s.features for s in masked_snapshots.values())
    assert "contact.euribor3m" in rows[0].features  # originals untouched


def test_protocols_split_and_controls_shuffle_labels() -> None:
    rows, _ = _synthetic_rows(300)
    train, test = PROTOCOLS["random_stratified"].split(rows, 7)
    assert len(train) + len(test) == 300 and 50 <= len(test) <= 70
    pit_train, pit_test = PROTOCOLS["point_in_time"].split(rows, 7)
    assert max(r.as_of for r in pit_train) < min(r.as_of for r in pit_test)
    recent_train, _ = PROTOCOLS["point_in_time_recent"].split(rows, 7)
    assert len(recent_train) < len(pit_train)
    shuffled = apply_post_split_control(train, "shuffled_labels", 7)
    assert sum(r.label for r in shuffled) == sum(r.label for r in train)
    assert [r.label for r in shuffled] != [r.label for r in train]


def test_evaluate_cell_reports_metrics_and_bootstrap_for_logistic_and_positive_rate() -> None:
    rows, snapshots = _synthetic_rows(600)
    train, test = PROTOCOLS["random_stratified"].split(rows, 7)
    logistic = MODEL_SPECS["logistic"].build(7, LAPTOP_PROFILE)
    cell = evaluate_cell(
        logistic, train, test, snapshots, precision_k=20, bootstrap_samples=50, seed=7
    )
    assert cell["status"] == "ok"
    assert float(cell["rocAuc"]) > 0.8  # type: ignore[arg-type]
    low, high = cell["ci95"]["rocAuc"]  # type: ignore[index]
    assert low <= float(cell["rocAuc"]) <= high  # type: ignore[arg-type]
    # Random targeting: a larger test partition keeps the chance-level AUC check stable.
    big_rows, big_snapshots = _synthetic_rows(2400, seed=5)
    big_train, big_test = PROTOCOLS["random_stratified"].split(big_rows, 7)
    baseline = evaluate_cell(
        PositiveRateScorer(7),
        big_train,
        big_test,
        big_snapshots,
        precision_k=20,
        bootstrap_samples=0,
        seed=7,
    )
    assert baseline["status"] == "ok"
    assert abs(float(baseline["rocAuc"]) - 0.5) < 0.1  # type: ignore[arg-type]
    assert baseline["ci95"] == {}


def test_aggregate_cells_averages_across_seeds() -> None:
    cells = [
        {"arm": "B1", "modelKey": "m", "seed": 1, "status": "ok", "rocAuc": 0.6, "prAuc": 0.2},
        {"arm": "B1", "modelKey": "m", "seed": 2, "status": "ok", "rocAuc": 0.8, "prAuc": 0.4},
        {"arm": "B1", "modelKey": "x", "seed": 1, "status": "unavailable", "detail": "no dep"},
    ]
    rows = {(r["arm"], r["modelKey"]): r for r in aggregate_cells(cells, ("arm", "modelKey"))}
    assert rows[("B1", "m")]["rocAucMean"] == pytest.approx(0.7)
    assert rows[("B1", "m")]["rocAucStd"] == pytest.approx(0.1)
    assert rows[("B1", "x")]["status"] == "unavailable"


def test_architecture_families_and_arms_are_nested() -> None:
    names = [
        "account.employee_count",
        "fit.score",
        "fit.hard_pass",
        "signal.sba_loan.loan_amount_usd.latest.max",
        "signal.sba_loan.loan_amount_usd.slope.mean",
        "signal.sba_loan.age_days.max",
        "signal.sba_loan.is_terminal.max",
        "signal.sba_loan.active_count",
        "signal.sba_loan.decayed_score",
        "signal.osha_injury_summary.max_instance_score",
        "signals.weighted_sum",
        "signals.distinct_types",
        "sequence.financed_then_operational_pressure.weighted",
        "story.sba_loan.phase_escalating",
    ]
    family_of = {
        "signal.sba_loan.loan_amount_usd.latest.max": "latest",
        "signal.sba_loan.loan_amount_usd.slope.mean": "reducers",
        "signal.sba_loan.age_days.max": "lifecycle",
        "signal.sba_loan.is_terminal.max": "stage",
        "signal.sba_loan.active_count": "active_count",
        "signal.sba_loan.decayed_score": "decay",
        "signals.weighted_sum": "decay",
        "signals.distinct_types": "rollup",
        "sequence.financed_then_operational_pressure.weighted": "sequence",
        "story.sba_loan.phase_escalating": "story",
        "fit.score": "fit",
        "account.employee_count": "account",
    }
    for name, family in family_of.items():
        assert ARCHITECTURE_FAMILIES[family](name), (name, family)
    assert ARCHITECTURE_FAMILIES["sba"]("signal.sba_loan.age_days.max")
    assert ARCHITECTURE_FAMILIES["sba"]("sequence.financed_then_operational_pressure.count")
    assert not ARCHITECTURE_FAMILIES["osha"]("signal.sba_loan.age_days.max")
    kept = {
        key: {name for name in names if ARCHITECTURE_ARMS[key].keep(name)}
        for key in (
            "B1_raw_latest",
            "B2_pit_recency",
            "B3_factsheet",
            "B4_decay",
            "B5_sequences",
            "B6_icp",
            "B7_story_tags",
        )
    }
    for earlier, later in pairwise(kept):
        assert kept[earlier] < kept[later], (earlier, later)
    assert "fit.score" not in kept["B5_sequences"] and "fit.score" in kept["B6_icp"]
    assert "story.sba_loan.phase_escalating" not in kept["B6_icp"]
    assert kept["B7_story_tags"] == set(names)


def test_shuffled_timestamps_keep_labels_and_per_account_timestamp_multisets() -> None:
    corpus = build_fixture_corpus(size=60, seed=11)
    horizon = 180
    before = pristine_labels(corpus.events, ARCHITECTURE_DATES, horizon)
    shuffled = shuffle_history_timestamps(corpus.events, seed=3)
    after = pristine_labels(shuffled, ARCHITECTURE_DATES, horizon)
    assert before == after
    cutoff = min(ARCHITECTURE_DATES)
    for account in {e.account_id for e in corpus.events}:
        original = sorted(
            e.available_at
            for e in corpus.events
            if e.account_id == account and e.available_at <= cutoff
        )
        permuted = sorted(
            e.available_at for e in shuffled if e.account_id == account and e.available_at <= cutoff
        )
        assert original == permuted
    assert len(shuffled) == len(corpus.events)


def test_new_instance_labels_ignore_later_stages_of_existing_instances() -> None:
    corpus = build_fixture_corpus(size=30, seed=2)
    template = corpus.events[0]
    as_of = ARCHITECTURE_DATES[0]

    def event(account: str, instance: str, occurred: datetime, event_id: str):  # type: ignore[no-untyped-def]
        return template.model_copy(
            update={
                "event_id": event_id,
                "account_id": account,
                "signal_instance_id": instance,
                "signal_type": "sba_loan",
                "occurred_at": occurred,
                "delivered_at": occurred,
                "available_at": occurred,
                "ingested_at": occurred,
            }
        )

    events = [
        # Existing loan: approved before as_of, disbursed inside the window.
        event("old", "loan-old", as_of - timedelta(days=200), "e1"),
        event("old", "loan-old", as_of + timedelta(days=30), "e2"),
        # New loan whose first event lands inside the window.
        event("new", "loan-new", as_of + timedelta(days=40), "e3"),
        # Loan far in the future: outside the 180-day horizon.
        event("late", "loan-late", as_of + timedelta(days=400), "e4"),
    ]
    any_event = pristine_labels(events, (as_of,), 180, label_mode="any_event")
    new_instance = pristine_labels(events, (as_of,), 180, label_mode="new_instance")
    key = as_of.isoformat()
    assert any_event[("old", key)] == 1 and new_instance[("old", key)] == 0
    assert any_event[("new", key)] == 1 and new_instance[("new", key)] == 1
    assert any_event[("late", key)] == 0 and new_instance[("late", key)] == 0
    with pytest.raises(ValueError):
        pristine_labels(events, (as_of,), 180, label_mode="anything")

    osha = template.model_copy(
        update={
            "event_id": "e5",
            "account_id": "old",
            "signal_instance_id": "osha-old",
            "signal_type": "osha_injury_summary",
        }
    )
    history = population_filter([*events, osha], "sba_history")
    assert history("old", as_of) and not history("new", as_of) and not history("late", as_of)
    assert history("new", as_of + timedelta(days=60))  # visible once the loan has appeared
    linked = population_filter([*events, osha], "linked")
    assert linked("old", as_of) and not linked("new", as_of)
    assert population_filter(events, "all")("anyone", as_of)
    with pytest.raises(ValueError):
        population_filter(events, "everyone")


def test_story_features_only_appear_when_requested() -> None:
    corpus = build_fixture_corpus(size=40, seed=5)
    registry = SignalRegistry.from_directory(ROOT / "signals")
    context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    plain = materialize_at(
        accounts=corpus.accounts,
        events=corpus.events,
        as_of=as_of,
        context=context,
        registry=registry,
    )
    tagged = materialize_at(
        accounts=corpus.accounts,
        events=corpus.events,
        as_of=as_of,
        context=context,
        registry=registry,
        include_story_features=True,
    )
    assert not any(name.startswith("story.") for s in plain for name in s.features)
    story_names = {name for s in tagged for name in s.features if name.startswith("story.")}
    assert {"story.phase_active.share", "story.phase_escalating.any"} <= story_names
    for plain_snapshot, tagged_snapshot in zip(plain, tagged, strict=True):
        shared = {k: v for k, v in tagged_snapshot.features.items() if not k.startswith("story.")}
        assert shared == plain_snapshot.features
