"""Versioned, language-neutral contracts shared by the engine, API, and UI."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SignalShape(StrEnum):
    NUMERIC_SERIES = "numeric_series"
    STATE_MACHINE = "state_machine"
    EVENT_BURST = "event_burst"
    DOCUMENT_VERSIONS = "document_versions"


class CoverageStatus(StrEnum):
    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"
    NOT_COLLECTED = "not_collected"
    NOT_APPLICABLE = "not_applicable"


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INDETERMINATE = "indeterminate"


class Action(StrEnum):
    CONTACT_NOW = "contact_now"
    WAIT = "wait"
    NEVER = "never"
    REVIEW = "review"


class EvidenceRef(FrozenModel):
    schema_version: Literal[1] = 1
    source_id: str
    source_record_id: str
    field: str | None = None
    observed_at: datetime | None = None


class NormalizedEvent(FrozenModel):
    schema_version: Literal[1] = 1
    event_id: str
    account_id: str
    signal_instance_id: str
    signal_type: str
    source_id: str
    source_record_id: str
    shape: SignalShape
    kind: str
    occurred_at: datetime
    delivered_at: datetime
    ingested_at: datetime
    available_at: datetime
    strength: float = Field(ge=0, le=1)
    payload: dict[str, Any]


class Fact(FrozenModel):
    schema_version: Literal[1] = 1
    fact_id: str
    name: str
    value: bool | float | int | str | None
    unit: str | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()


class FactSheet(FrozenModel):
    schema_version: Literal[1] = 1
    factsheet_id: str
    signal_instance_id: str
    signal_type: str
    account_id: str
    shape: SignalShape
    as_of: datetime
    coverage: CoverageStatus
    identity: dict[str, str | int | float | None]
    current_state: dict[str, str | int | float | bool | None]
    facts: tuple[Fact, ...]
    semantic_hash: str


class StoryCard(FrozenModel):
    schema_version: Literal[1] = 1
    story_card_id: str
    scope: Literal["signal", "account"]
    subject_id: str
    story_type: str
    phase: str
    confidence: float = Field(ge=0, le=1)
    summary: str
    implications: tuple[str, ...] = ()
    evidence_fact_ids: tuple[str, ...]
    semantic_hash: str
    generator: Literal["deterministic", "llm"]
    model: str | None = None
    prompt_version: str | None = None
    created_at: datetime


class ProductSpec(FrozenModel):
    schema_version: Literal[1] = 1
    product_version: str
    name: str
    website: HttpUrl
    description: str
    capabilities: tuple[str, ...]
    problems_solved: tuple[str, ...]


class IcpSpec(FrozenModel):
    schema_version: Literal[1] = 1
    icp_version: str
    name: str
    industries: tuple[str, ...]
    naics_prefixes: tuple[str, ...] = ()
    geographies: tuple[str, ...] = ()
    employee_min: int | None = Field(default=None, ge=1)
    employee_max: int | None = Field(default=None, ge=1)
    operational_traits: tuple[str, ...] = ()
    buyer_titles: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()


class GoalSpec(FrozenModel):
    schema_version: Literal[1] = 1
    goal_version: str
    name: str
    objective: str
    target_outcome: str
    horizon_days: int = Field(ge=1, le=730)


class SignalPolicyEntry(FrozenModel):
    signal_type: str
    applicability_naics_prefixes: tuple[str, ...] = ()
    initial_weight: float = Field(ge=0, le=1)
    half_life_days: float = Field(gt=0)
    pinned: bool = False


class SequenceRule(FrozenModel):
    rule_id: str
    ordered_signal_types: tuple[str, ...] = Field(min_length=2)
    max_gap_days: int = Field(ge=1, le=3_650)
    weight: float = Field(ge=0, le=1)
    description: str


class SignalPolicy(FrozenModel):
    schema_version: Literal[1] = 1
    signal_policy_version: str
    entries: tuple[SignalPolicyEntry, ...]
    sequence_rules: tuple[SequenceRule, ...] = ()


class CorpusSourceManifest(FrozenModel):
    source_id: str
    landing_page: HttpUrl
    retrieval_url: HttpUrl | None = None
    license: str
    retrieved_at: datetime
    content_sha256: str
    local_path: str
    bytes: int = Field(ge=0)


class CorpusManifest(FrozenModel):
    schema_version: Literal[1] = 1
    corpus_version: str
    created_at: datetime
    sources: tuple[CorpusSourceManifest, ...]
    manifest_hash: str


class AccountRecord(FrozenModel):
    schema_version: Literal[1] = 1
    account_id: str
    name: str
    naics_code: str | None = None
    city: str | None = None
    state: str | None = None
    employee_count: int | None = Field(default=None, ge=0)
    source_identifiers: dict[str, str]
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ExperimentSpec(FrozenModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    engine_version: str
    feature_version: str
    signal_contracts: dict[str, str]
    product: ProductSpec
    icp: IcpSpec
    goal: GoalSpec
    signal_policy: SignalPolicy
    corpus_version: str
    as_of: datetime
    model_ids: tuple[str, ...]
    capacity: int = Field(ge=1, le=10_000)
    max_accounts: int = Field(default=5_000, ge=10, le=1_000_000)
    candidate_multiplier: int = Field(default=3, ge=1, le=20)
    random_seed: int = 7
    max_llm_story_cards: int = Field(default=0, ge=0, le=10_000)


class AccountSnapshot(FrozenModel):
    schema_version: Literal[1] = 1
    account_id: str
    as_of: datetime
    icp_version: str
    goal_version: str
    features: dict[str, float]
    coverage: dict[str, CoverageStatus]
    evidence_fact_ids: tuple[str, ...]
    snapshot_hash: str


class ModelScore(FrozenModel):
    schema_version: Literal[1] = 1
    model_id: str
    account_id: str
    score: float = Field(ge=0, le=1)
    uncertainty: float | None = Field(default=None, ge=0)
    top_factors: tuple[tuple[str, float], ...] = ()
    calibrated: bool = False


class TrainingRow(FrozenModel):
    schema_version: Literal[1] = 1
    account_id: str
    as_of: datetime
    features: dict[str, float]
    label: int = Field(ge=0, le=1)
    label_kind: Literal["synthetic_oracle", "public_future_event", "real_outcome"]


class BenchmarkResult(FrozenModel):
    schema_version: Literal[1] = 1
    model_id: str
    label_kind: str
    train_rows: int = Field(ge=0)
    test_rows: int = Field(ge=0)
    pr_auc: float | None
    roc_auc: float | None
    brier_score: float | None
    precision_at_k: float | None
    fit_seconds: float = Field(ge=0)
    score_seconds: float = Field(ge=0)
    status: Literal["ok", "unavailable", "failed"]
    detail: str | None = None


class PortfolioWeight(FrozenModel):
    schema_version: Literal[1] = 1
    signal_type: str
    observations: int = Field(ge=0)
    positives: int = Field(ge=0)
    baseline_rate: float = Field(ge=0, le=1)
    exposed_rate: float = Field(ge=0, le=1)
    smoothed_lift: float = Field(ge=0)
    previous_weight: float = Field(ge=0, le=1)
    proposed_weight: float = Field(ge=0.05, le=1)


class ActionEvaluation(FrozenModel):
    schema_version: Literal[1] = 1
    action: Action
    wait_days: int = Field(default=0, ge=0, le=365)
    response_probability: float = Field(ge=0, le=1)
    expected_value: float


class Recommendation(FrozenModel):
    schema_version: Literal[1] = 1
    account_id: str
    rank: int = Field(ge=1)
    action: Action
    best_window_start: date | None
    best_window_end: date | None
    opportunity_score: float = Field(ge=0, le=1)
    icp_fit_score: float = Field(default=0, ge=0, le=1)
    signal_relevance_score: float = Field(default=0, ge=0, le=1)
    proxy_event_score: float | None = Field(default=None, ge=0, le=1)
    data_confidence_score: float = Field(default=0, ge=0, le=1)
    model_disagreement: bool = False
    decision_status: Literal[
        "actionable",
        "human_review",
        "insufficient_evidence",
        "research_only",
    ] = "research_only"
    model_scores: tuple[ModelScore, ...]
    action_evaluations: tuple[ActionEvaluation, ...]
    reason: str
    evidence_fact_ids: tuple[str, ...]
    story_card: StoryCard | None = None


class RecommendRequest(FrozenModel):
    schema_version: Literal[1] = 1
    workspace_id: str
    icp_version_id: str
    goal_version_id: str
    as_of: datetime
    capacity: int = Field(ge=1, le=10_000)
    idempotency_key: str = Field(min_length=1, max_length=128)


class RecommendResponse(FrozenModel):
    schema_version: Literal[1] = 1
    run_id: str
    state: RunState
    recommendations: tuple[Recommendation, ...]


class RunEvent(FrozenModel):
    schema_version: Literal[1] = 1
    run_id: str
    seq: int = Field(ge=1)
    kind: str
    created_at: datetime
    payload: dict[str, Any]


class ExperimentRun(FrozenModel):
    schema_version: Literal[1] = 1
    run_id: str
    spec: ExperimentSpec
    spec_hash: str
    state: RunState
    created_at: datetime
    started_at: datetime | None = None
    terminal_at: datetime | None = None
    error_code: str | None = None
