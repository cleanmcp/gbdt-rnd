"""Durable, replayable experiment coordinator with bounded step events."""

from __future__ import annotations

import asyncio
import os
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .config import DecisionContext
from .contracts import (
    BenchmarkResult,
    ExperimentRun,
    ExperimentSpec,
    FactSheet,
    ModelScore,
    NormalizedEvent,
    RunState,
)
from .facts import build_factsheet
from .fixtures import FixtureCorpus, build_fixture_corpus
from .hashing import sha256_json
from .models import (
    HeuristicScorer,
    LightGbmScorer,
    Scorer,
    TabPfnScorer,
    benchmark,
)
from .portfolio import propose_portfolio_weights
from .signal_registry import SignalRegistry
from .simulation import build_recommendation, recommendation_priority
from .store import ExperimentStore
from .stories import (
    OpenAiNarrator,
    StoryBudget,
    TemplateNarrator,
    account_story,
    deterministic_signal_story,
)
from .training import build_training_materialization, materialize_at


def _story_fact_payload(sheets: list[FactSheet]) -> list[dict[str, object]]:
    def strength(sheet: FactSheet) -> float:
        for fact in sheet.facts:
            if fact.name == "latest_strength" and isinstance(fact.value, (float, int)):
                return float(fact.value)
        return 0

    payload: list[dict[str, object]] = []
    for sheet in sorted(sheets, key=strength, reverse=True)[:8]:
        payload.append(
            {
                "signal_instance_id": sheet.signal_instance_id,
                "signal_type": sheet.signal_type,
                "current_state": sheet.current_state,
                "facts": [
                    {
                        "fact_id": fact.fact_id,
                        "name": fact.name,
                        "value": fact.value,
                        "unit": fact.unit,
                    }
                    for fact in sheet.facts[:12]
                ],
            }
        )
    return payload


def build_experiment_spec(
    context: DecisionContext,
    *,
    as_of: datetime,
    corpus_version: str = "fixture-manufacturing-v1",
    capacity: int = 25,
    max_accounts: int = 5_000,
    signal_contracts: dict[str, str] | None = None,
    model_ids: tuple[str, ...] = (
        "heuristic-v1",
        "lightgbm-v1",
        "tabpfn-local-v2",
    ),
    max_llm_story_cards: int = 25,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=str(uuid4()),
        engine_version="signal-engine-0.2.0",
        feature_version="account-features-v1",
        signal_contracts=signal_contracts or {},
        product=context.product,
        icp=context.icp,
        goal=context.goal,
        signal_policy=context.signal_policy,
        corpus_version=corpus_version,
        as_of=as_of,
        model_ids=model_ids,
        capacity=capacity,
        max_accounts=max_accounts,
        max_llm_story_cards=max_llm_story_cards,
    )


class RunCoordinator:
    def __init__(
        self,
        *,
        store: ExperimentStore,
        registry: SignalRegistry,
        fixture: FixtureCorpus | None = None,
        artifact_dir: Path | None = None,
    ):
        self.store = store
        self.registry = registry
        self.fixture = fixture
        self.artifact_dir = artifact_dir
        self._tasks: set[asyncio.Task[None]] = set()

    def bootstrap_fixture(self, size: int = 180) -> FixtureCorpus:
        fixture = build_fixture_corpus(size=size)
        self.store.upsert_accounts(fixture.accounts)
        self.store.append_events(fixture.events)
        self.fixture = fixture
        return fixture

    def reconcile_interrupted_runs(self) -> int:
        reconciled = 0
        for run in self.store.running_runs():
            terminal = datetime.now(UTC)
            self.store.save_run(
                run.model_copy(
                    update={
                        "state": RunState.INDETERMINATE,
                        "terminal_at": terminal,
                        "error_code": "PROCESS_RESTARTED",
                    }
                )
            )
            self.store.append_run_event(
                run.run_id,
                "run.indeterminate",
                {
                    "errorCode": "PROCESS_RESTARTED",
                    "terminalAt": terminal.isoformat(),
                },
            )
            reconciled += 1
        return reconciled

    def request(self, spec: ExperimentSpec) -> tuple[ExperimentRun, bool]:
        current_contracts = self.registry.manifest()
        if spec.signal_contracts and spec.signal_contracts != current_contracts:
            raise ValueError("signal registry contract hash drift")
        registered = {definition.signal_type for definition in self.registry.all()}
        requested_signals = {entry.signal_type for entry in spec.signal_policy.entries} | {
            signal_type
            for rule in spec.signal_policy.sequence_rules
            for signal_type in rule.ordered_signal_types
        }
        unknown_signals = requested_signals - registered
        if unknown_signals:
            raise ValueError(f"signal policy references unknown signals: {sorted(unknown_signals)}")
        known_models = {"heuristic-v1", "lightgbm-v1", "tabpfn-local-v2"}
        unknown_models = set(spec.model_ids) - known_models
        if unknown_models:
            raise ValueError(f"unknown model ids: {sorted(unknown_models)}")
        payload = spec.model_dump(mode="json")
        payload["experiment_id"] = None
        spec_hash = sha256_json(payload)
        existing = self.store.run_by_spec_hash(spec_hash)
        if existing:
            return existing, False
        now = datetime.now(UTC)
        run = ExperimentRun(
            run_id=str(uuid4()),
            spec=spec,
            spec_hash=spec_hash,
            state=RunState.QUEUED,
            created_at=now,
        )
        self.store.create_run(run)
        self.store.append_run_event(
            run.run_id,
            "run.queued",
            {
                "specHash": spec_hash,
                "corpusVersion": spec.corpus_version,
                "modelIds": list(spec.model_ids),
            },
        )
        return run, True

    def start(self, run_id: str) -> None:
        task = asyncio.create_task(self._execute_in_worker(run_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute_in_worker(self, run_id: str) -> None:
        await asyncio.to_thread(lambda: asyncio.run(self.execute(run_id)))

    def _scorer(self, model_id: str, seed: int) -> Scorer:
        if model_id == "heuristic-v1":
            return HeuristicScorer()
        if model_id == "lightgbm-v1":
            return LightGbmScorer(seed)
        if model_id == "tabpfn-local-v2":
            return TabPfnScorer(seed)
        raise ValueError(f"unknown model: {model_id}")

    async def execute(self, run_id: str) -> None:
        run = self.store.run(run_id)
        if run is None or run.state != RunState.QUEUED:
            return
        started = datetime.now(UTC)
        run = run.model_copy(update={"state": RunState.RUNNING, "started_at": started})
        self.store.save_run(run)
        self.store.append_run_event(run_id, "run.started", {"startedAt": started.isoformat()})
        try:
            if not self.store.accounts():
                if self.fixture is None:
                    raise RuntimeError(
                        "selected corpus database is empty; import a public source first"
                    )
                self.bootstrap_fixture(size=len(self.fixture.accounts))
            all_accounts = self.store.accounts()
            accounts = all_accounts[: run.spec.max_accounts]
            selected_account_ids = {account.account_id for account in accounts}
            events = [
                event for event in self.store.events() if event.account_id in selected_account_ids
            ]
            context = DecisionContext(
                product=run.spec.product,
                icp=run.spec.icp,
                goal=run.spec.goal,
                signal_policy=run.spec.signal_policy,
            )
            self.store.append_run_event(
                run_id,
                "data.loaded",
                {
                    "accounts": len(accounts),
                    "totalAccounts": len(all_accounts),
                    "events": len(events),
                    "sampled": len(accounts) < len(all_accounts),
                },
            )
            await asyncio.sleep(0)

            grouped_events: dict[str, list[NormalizedEvent]] = defaultdict(list)
            for event in events:
                if event.available_at <= run.spec.as_of:
                    grouped_events[event.signal_instance_id].append(event)
            factsheets: list[FactSheet] = []
            cache_hits = 0
            for instance_events in grouped_events.values():
                sheet = build_factsheet(
                    self.registry.get(instance_events[0].signal_type),
                    instance_events,
                    run.spec.as_of,
                )
                factsheets.append(sheet)
                self.store.save_factsheet(sheet)
                existing = self.store.story_card(
                    "signal",
                    sheet.signal_instance_id,
                    sheet.semantic_hash,
                    "deterministic",
                )
                if existing:
                    cache_hits += 1
                else:
                    self.store.save_story_card(deterministic_signal_story(sheet))
            self.store.append_run_event(
                run_id,
                "factsheets.materialized",
                {
                    "factsheets": len(factsheets),
                    "signalStoryCacheHits": cache_hits,
                    "signalStoryWrites": len(factsheets) - cache_hits,
                },
            )
            await asyncio.sleep(0)

            snapshots = materialize_at(
                accounts=accounts,
                events=events,
                as_of=run.spec.as_of,
                context=context,
                registry=self.registry,
            )
            eligible = [
                snapshot
                for snapshot in snapshots
                if snapshot.features.get("fit.hard_pass", 0) == 1
                and snapshot.features.get("signals.distinct_types", 0) > 0
            ]
            self.store.append_run_event(
                run_id,
                "accounts.consolidated",
                {
                    "snapshots": len(snapshots),
                    "eligible": len(eligible),
                    "featureCount": max(
                        (len(snapshot.features) for snapshot in snapshots), default=0
                    ),
                },
            )
            await asyncio.sleep(0)

            dates = (
                datetime(2024, 6, 30, tzinfo=UTC),
                datetime(2024, 12, 31, tzinfo=UTC),
                datetime(2025, 3, 31, tzinfo=UTC),
                datetime(2025, 6, 30, tzinfo=UTC),
                datetime(2025, 9, 30, tzinfo=UTC),
                datetime(2025, 12, 31, tzinfo=UTC),
            )
            label_kind = "synthetic_oracle" if self.fixture is not None else "public_future_event"
            training = build_training_materialization(
                accounts=accounts,
                events=events,
                dates=dates,
                context=context,
                registry=self.registry,
                latent_opportunity=(
                    self.fixture.latent_opportunity if self.fixture is not None else None
                ),
                label_kind=label_kind,
            )
            snapshots_by_key = {
                (snapshot.account_id, snapshot.as_of.isoformat()): snapshot
                for snapshot in training.snapshots
            }
            self.store.append_run_event(
                run_id,
                "training_set.built",
                {
                    "rows": len(training.rows),
                    "labelKind": label_kind,
                    "positiveRows": sum(row.label for row in training.rows),
                },
            )
            await asyncio.sleep(0)
            portfolio = propose_portfolio_weights(
                training.rows,
                run.spec.signal_policy,
            )
            self.store.append_run_event(
                run_id,
                "portfolio.proposed",
                {
                    "weights": [weight.model_dump(mode="json") for weight in portfolio],
                    "activation": "proposal_only_next_policy_version",
                },
            )
            await asyncio.sleep(0)

            scores_by_account: dict[str, list[ModelScore]] = defaultdict(list)
            benchmark_results: list[BenchmarkResult] = []
            for model_id in run.spec.model_ids:
                benchmark_scorer = self._scorer(model_id, run.spec.random_seed)
                result = benchmark(
                    benchmark_scorer,
                    training.rows,
                    snapshots_by_key,
                    precision_k=run.spec.capacity,
                )
                benchmark_results.append(result)
                self.store.save_benchmark(run_id, result)
                scorer = self._scorer(model_id, run.spec.random_seed)
                try:
                    scorer.fit(training.rows)
                    for score in scorer.score(eligible):
                        scores_by_account[score.account_id].append(score)
                    if isinstance(scorer, LightGbmScorer) and self.artifact_dir:
                        artifact = scorer.save_artifact(
                            self.artifact_dir / run_id / scorer.model_id
                        )
                        self.store.append_run_event(
                            run_id,
                            "model.artifact_written",
                            {"modelId": scorer.model_id, **artifact},
                        )
                except Exception as error:
                    self.store.append_run_event(
                        run_id,
                        "model.unavailable",
                        {
                            "modelId": model_id,
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
            self.store.append_run_event(
                run_id,
                "models.benchmarked",
                {"results": [result.model_dump(mode="json") for result in benchmark_results]},
            )
            await asyncio.sleep(0)
            ranked = sorted(
                (
                    (
                        recommendation_priority(snapshot, tuple(scores)),
                        snapshot,
                        tuple(scores),
                    )
                    for snapshot in eligible
                    if (scores := scores_by_account.get(snapshot.account_id))
                ),
                key=lambda item: item[0],
                reverse=True,
            )[: run.spec.capacity * run.spec.candidate_multiplier]
            recommendations = [
                build_recommendation(index + 1, snapshot, scores, label_kind)
                for index, (_, snapshot, scores) in enumerate(ranked[: run.spec.capacity])
            ]

            narration_model = os.getenv("SIGNAL_ENGINE_NARRATION_MODEL")
            narrator = (
                OpenAiNarrator(narration_model)
                if narration_model and os.getenv("OPENAI_API_KEY")
                else TemplateNarrator()
            )
            budget = StoryBudget(run.spec.max_llm_story_cards)
            sheets_by_account: dict[str, list[FactSheet]] = defaultdict(list)
            for sheet in factsheets:
                sheets_by_account[sheet.account_id].append(sheet)
            enriched = []
            snapshot_by_id = {snapshot.account_id: snapshot for snapshot in eligible}
            for recommendation in recommendations:
                snapshot = snapshot_by_id[recommendation.account_id]
                fact_payload = _story_fact_payload(sheets_by_account[recommendation.account_id])
                try:
                    card = await account_story(
                        store=self.store,
                        provider=narrator,
                        budget=budget,
                        product=run.spec.product,
                        icp=run.spec.icp,
                        snapshot=snapshot,
                        fact_payload=fact_payload,
                    )
                except Exception as error:
                    self.store.append_run_event(
                        run_id,
                        "story_card.fallback",
                        {
                            "accountId": recommendation.account_id,
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                    card = await account_story(
                        store=self.store,
                        provider=TemplateNarrator(),
                        budget=StoryBudget(1),
                        product=run.spec.product,
                        icp=run.spec.icp,
                        snapshot=snapshot,
                        fact_payload=fact_payload,
                    )
                enriched.append(recommendation.model_copy(update={"story_card": card}))
            self.store.save_recommendations(run_id, enriched)
            self.store.append_run_event(
                run_id,
                "recommendations.created",
                {
                    "candidateAccounts": len(ranked),
                    "recommendations": len(enriched),
                    "storyCardsGeneratedOrRead": sum(
                        recommendation.story_card is not None for recommendation in enriched
                    ),
                    "llmCalls": (budget.used if isinstance(narrator, OpenAiNarrator) else 0),
                },
            )

            terminal = datetime.now(UTC)
            completed = run.model_copy(
                update={"state": RunState.COMPLETED, "terminal_at": terminal}
            )
            self.store.save_run(completed)
            self.store.append_run_event(
                run_id,
                "run.completed",
                {
                    "terminalAt": terminal.isoformat(),
                    "recommendations": len(enriched),
                },
            )
        except Exception as error:
            terminal = datetime.now(UTC)
            failed = run.model_copy(
                update={
                    "state": RunState.FAILED,
                    "terminal_at": terminal,
                    "error_code": type(error).__name__,
                }
            )
            self.store.save_run(failed)
            self.store.append_run_event(
                run_id,
                "run.failed",
                {
                    "errorCode": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(limit=8),
                },
            )


def default_coordinator(root: Path) -> RunCoordinator:
    mode = os.getenv("SIGNAL_ENGINE_CORPUS_MODE", "fixture")
    configured_path = os.getenv("SIGNAL_ENGINE_DB_PATH")
    database_path = Path(configured_path) if configured_path else root / "data" / f"{mode}.duckdb"
    store = ExperimentStore(database_path)
    registry = SignalRegistry.from_directory(root / "signals")
    coordinator = RunCoordinator(
        store=store,
        registry=registry,
        artifact_dir=root / "artifacts",
    )
    coordinator.reconcile_interrupted_runs()
    if mode == "fixture":
        fixture = build_fixture_corpus()
        coordinator.fixture = fixture
        if not store.accounts():
            store.upsert_accounts(fixture.accounts)
            store.append_events(fixture.events)
    return coordinator
