"""Real, anonymized outcome benchmarks that are deliberately separate from world signals."""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import AccountSnapshot, BenchmarkResult, TrainingRow
from .hashing import sha256_file, sha256_json
from .models import (
    GPU_TABPFN_CONFIG,
    LAPTOP_TABPFN_CONFIG,
    HeuristicScorer,
    LightGbmScorer,
    Scorer,
    TabPfnConfig,
    TabPfnScorer,
    benchmark,
    hardware_summary,
)


@dataclass(frozen=True)
class BenchmarkProfile:
    """Where and how heavily the models run; the data, labels, and splits never change."""

    name: str
    description: str
    tabpfn: TabPfnConfig

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "tabpfn": self.tabpfn.describe(),
            "lightgbmDevice": "cpu",
        }


LAPTOP_PROFILE = BenchmarkProfile(
    name="laptop-cpu",
    description="Local CPU configuration: TabPFN capped at 600 training rows, 2 estimators.",
    tabpfn=LAPTOP_TABPFN_CONFIG,
)
GPU_PROFILE = BenchmarkProfile(
    name="gpu-full",
    description=(
        "CUDA configuration: TabPFN uses the full point-in-time training partition "
        "(capped at 50,000 rows) with 8 estimators; LightGBM stays on CPU."
    ),
    tabpfn=GPU_TABPFN_CONFIG,
)
PROFILES: dict[str, BenchmarkProfile] = {"laptop": LAPTOP_PROFILE, "gpu": GPU_PROFILE}


def resolve_profile(
    name: str,
    *,
    tabpfn_max_rows: int | None = None,
    tabpfn_estimators: int | None = None,
    tabpfn_device: str | None = None,
) -> BenchmarkProfile:
    """Look up a named profile and apply explicit TabPFN overrides (recorded in reports)."""
    try:
        profile = PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown benchmark profile {name!r}; use one of {sorted(PROFILES)}"
        ) from error
    requested: dict[str, object] = {}
    if tabpfn_max_rows is not None:
        requested["max_training_rows"] = tabpfn_max_rows if tabpfn_max_rows > 0 else None
    if tabpfn_estimators is not None:
        requested["n_estimators"] = tabpfn_estimators
    if tabpfn_device is not None:
        requested["device"] = tabpfn_device
    current = profile.tabpfn.describe()
    updates = {key: value for key, value in requested.items() if current[key] != value}
    if not updates:
        return profile
    return replace(
        profile,
        name=f"{profile.name}-custom",
        tabpfn=replace(profile.tabpfn, **updates),  # type: ignore[arg-type]
    )


def _classification_scorers(profile: BenchmarkProfile, random_seed: int) -> tuple[Scorer, ...]:
    return (
        HeuristicScorer(),
        LightGbmScorer(random_seed),
        TabPfnScorer(random_seed, config=profile.tabpfn),
    )


def _execution_envelope(
    profile: BenchmarkProfile,
    scorers: Sequence[Scorer],
) -> dict[str, object]:
    return {
        "generatedAt": datetime.now(UTC).isoformat(),
        "executionProfile": profile.describe(),
        "hardware": hardware_summary(),
        "modelRuntimes": {
            scorer.model_id: runtime for scorer in scorers if (runtime := scorer.runtime_info())
        },
    }


@dataclass(frozen=True)
class OlistData:
    rows: list[TrainingRow]
    snapshots_by_key: dict[tuple[str, str], AccountSnapshot]
    leads: int
    won: int
    average_days_to_close: float


@dataclass(frozen=True)
class BinaryOutcomeData:
    rows: list[TrainingRow]
    snapshots_by_key: dict[tuple[str, str], AccountSnapshot]
    observations: int
    positives: int


def _archive_csv(archive: zipfile.ZipFile, name: str) -> pl.DataFrame:
    with archive.open(name) as source:
        return pl.read_csv(io.BytesIO(source.read()))


def load_olist(path: Path) -> OlistData:
    with zipfile.ZipFile(path) as archive:
        leads = _archive_csv(archive, "olist_marketing_qualified_leads_dataset.csv")
        deals = _archive_csv(archive, "olist_closed_deals_dataset.csv")

    won_dates = {
        str(row["mql_id"]): datetime.fromisoformat(str(row["won_date"])).replace(tzinfo=UTC)
        for row in deals.iter_rows(named=True)
        if row.get("won_date")
    }
    origins = sorted(
        {
            str(origin).strip().lower()
            for origin in leads.get_column("origin").drop_nulls().to_list()
        }
    )
    rows: list[TrainingRow] = []
    snapshots: dict[tuple[str, str], AccountSnapshot] = {}
    close_days = []
    for lead in leads.iter_rows(named=True):
        account_id = str(lead["mql_id"])
        as_of = datetime.fromisoformat(str(lead["first_contact_date"])).replace(tzinfo=UTC)
        origin = str(lead.get("origin") or "unknown").strip().lower()
        landing_page = str(lead.get("landing_page_id") or "unknown")
        bucket = int(hashlib.sha256(landing_page.encode()).hexdigest()[:8], 16) % 32
        day_of_year = as_of.timetuple().tm_yday
        features = {
            "fit.score": 1.0,
            "contact.month": float(as_of.month),
            "contact.weekday": float(as_of.weekday()),
            "contact.year_progress_sin": math.sin(2 * math.pi * day_of_year / 365),
            "contact.year_progress_cos": math.cos(2 * math.pi * day_of_year / 365),
            f"origin.{origin}": 1.0,
            f"landing_page.bucket_{bucket:02d}": 1.0,
        }
        for known_origin in origins:
            features.setdefault(f"origin.{known_origin}", 0.0)
        snapshot_payload = {
            "account_id": account_id,
            "as_of": as_of,
            "features": features,
        }
        snapshot = AccountSnapshot(
            account_id=account_id,
            as_of=as_of,
            icp_version="olist-merchant-acquisition-v1",
            goal_version="closed-deal-v1",
            features=features,
            coverage={},
            evidence_fact_ids=(),
            snapshot_hash=sha256_json(snapshot_payload),
        )
        label = int(account_id in won_dates)
        if label:
            close_days.append((won_dates[account_id] - as_of).days)
        row = TrainingRow(
            account_id=account_id,
            as_of=as_of,
            features=features,
            label=label,
            label_kind="real_outcome",
        )
        rows.append(row)
        snapshots[(account_id, as_of.isoformat())] = snapshot
    return OlistData(
        rows=rows,
        snapshots_by_key=snapshots,
        leads=len(rows),
        won=sum(row.label for row in rows),
        average_days_to_close=sum(close_days) / len(close_days),
    )


def run_olist_benchmark(
    path: Path,
    report_path: Path,
    *,
    random_seed: int = 7,
    profile: BenchmarkProfile = LAPTOP_PROFILE,
) -> dict[str, object]:
    data = load_olist(path)
    scorers = _classification_scorers(profile, random_seed)
    results: list[BenchmarkResult] = [
        benchmark(
            scorer,
            data.rows,
            data.snapshots_by_key,
            precision_k=100,
        )
        for scorer in scorers
    ]
    report: dict[str, object] = {
        "schemaVersion": 1,
        "dataset": "Olist Marketing Funnel",
        "source": "https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist",
        "license": "Kaggle dataset terms; verify before commercial use",
        "contentSha256": sha256_file(path),
        "leads": data.leads,
        "won": data.won,
        "winRate": data.won / data.leads,
        "averageDaysToClose": data.average_days_to_close,
        "target": "closed deal",
        "featureBoundary": (
            "Only fields available in the marketing-qualified-lead file are features; "
            "closed-deal fields are labels only."
        ),
        "benchmarks": [result.model_dump(mode="json") for result in results],
        **_execution_envelope(profile, scorers),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _slug(value: object) -> str:
    return (
        str(value or "unknown")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def load_uci_bank_marketing(path: Path) -> BinaryOutcomeData:
    with (
        zipfile.ZipFile(path) as outer,
        zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip"))) as inner,
        inner.open("bank-additional/bank-additional-full.csv") as source,
    ):
        frame = pl.read_csv(
            io.BytesIO(source.read()),
            separator=";",
            infer_schema_length=10_000,
        )

    excluded = {"y", "duration"}
    categorical = [
        column
        for column, dtype in frame.schema.items()
        if dtype == pl.String and column not in excluded
    ]
    vocabularies = {
        column: sorted({_slug(value) for value in frame.get_column(column).to_list()})
        for column in categorical
    }
    numeric = [
        column for column in frame.columns if column not in excluded and column not in categorical
    ]
    start = datetime(2008, 5, 1, tzinfo=UTC)
    span_seconds = (datetime(2010, 11, 30, tzinfo=UTC) - start).total_seconds()
    rows: list[TrainingRow] = []
    snapshots: dict[tuple[str, str], AccountSnapshot] = {}
    total = frame.height
    for index, source_row in enumerate(frame.iter_rows(named=True)):
        account_id = f"uci-bank-contact-{index:05d}"
        as_of = start + timedelta(seconds=span_seconds * index / max(total - 1, 1))
        features = {
            f"contact.{column}": float(source_row[column])
            for column in numeric
            if source_row[column] is not None
        }
        features["fit.score"] = 1.0
        for column in categorical:
            observed = _slug(source_row[column])
            for category in vocabularies[column]:
                features[f"contact.{column}.{category}"] = float(category == observed)
        payload = {"account_id": account_id, "as_of": as_of, "features": features}
        snapshot = AccountSnapshot(
            account_id=account_id,
            as_of=as_of,
            icp_version="uci-bank-contact-v1",
            goal_version="term-deposit-response-v1",
            features=features,
            coverage={},
            evidence_fact_ids=(),
            snapshot_hash=sha256_json(payload),
        )
        row = TrainingRow(
            account_id=account_id,
            as_of=as_of,
            features=features,
            label=int(str(source_row["y"]).lower() == "yes"),
            label_kind="real_outcome",
        )
        rows.append(row)
        snapshots[(account_id, as_of.isoformat())] = snapshot
    return BinaryOutcomeData(
        rows=rows,
        snapshots_by_key=snapshots,
        observations=len(rows),
        positives=sum(row.label for row in rows),
    )


def run_uci_bank_benchmark(
    path: Path,
    report_path: Path,
    *,
    random_seed: int = 7,
    profile: BenchmarkProfile = LAPTOP_PROFILE,
) -> dict[str, object]:
    data = load_uci_bank_marketing(path)
    scorers = _classification_scorers(profile, random_seed)
    results = [
        benchmark(
            scorer,
            data.rows,
            data.snapshots_by_key,
            precision_k=250,
        )
        for scorer in scorers
    ]
    report: dict[str, object] = {
        "schemaVersion": 1,
        "dataset": "UCI Bank Marketing",
        "source": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "contentSha256": sha256_file(path),
        "observations": data.observations,
        "positives": data.positives,
        "responseRate": data.positives / data.observations,
        "target": "subscribed after direct-marketing contact",
        "featureBoundary": (
            "Call duration is excluded because it is only known after contact. "
            "Client attributes, campaign history, channel, calendar, prior outcome, "
            "and public macro variables are retained."
        ),
        "limitations": (
            "Rows are chronologically ordered but exact years are not present in the CSV; "
            "the benchmark preserves source order on an approximate calendar. No stable "
            "client identifier is published, so repeated clients cannot be grouped."
        ),
        "benchmarks": [result.model_dump(mode="json") for result in results],
        **_execution_envelope(profile, scorers),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


@dataclass(frozen=True)
class HillstromData:
    snapshots: list[AccountSnapshot]
    rows: list[TrainingRow]
    treatment: list[int]
    conversion: list[int]
    spend: list[float]


def load_hillstrom(path: Path) -> HillstromData:
    frame = pl.read_csv(path)
    categorical = ("history_segment", "zip_code", "channel")
    vocabularies = {
        column: sorted({_slug(value) for value in frame.get_column(column).to_list()})
        for column in categorical
    }
    numeric = ("recency", "history", "mens", "womens", "newbie")
    snapshots = []
    rows = []
    treatment = []
    conversion = []
    spend = []
    as_of = datetime(2008, 3, 20, tzinfo=UTC)
    for index, source_row in enumerate(frame.iter_rows(named=True)):
        account_id = f"hillstrom-customer-{index:05d}"
        features = {f"customer.{column}": float(source_row[column]) for column in numeric}
        features["fit.score"] = 1.0
        for column in categorical:
            observed = _slug(source_row[column])
            for category in vocabularies[column]:
                features[f"customer.{column}.{category}"] = float(category == observed)
        payload = {"account_id": account_id, "features": features}
        snapshot = AccountSnapshot(
            account_id=account_id,
            as_of=as_of,
            icp_version="hillstrom-retail-v1",
            goal_version="email-conversion-uplift-v1",
            features=features,
            coverage={},
            evidence_fact_ids=(),
            snapshot_hash=sha256_json(payload),
        )
        label = int(source_row["conversion"])
        snapshots.append(snapshot)
        rows.append(
            TrainingRow(
                account_id=account_id,
                as_of=as_of,
                features=features,
                label=label,
                label_kind="real_outcome",
            )
        )
        treatment.append(int(str(source_row["segment"]) != "No E-Mail"))
        conversion.append(label)
        spend.append(float(source_row["spend"]))
    return HillstromData(snapshots, rows, treatment, conversion, spend)


def _observed_rate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    indices: np.ndarray,
    treated: bool,
) -> float:
    selected = indices[treatment[indices] == int(treated)]
    return float(outcomes[selected].mean()) if len(selected) else 0.0


def _uplift_model_report(
    model_id: str,
    factory: Callable[[int], Scorer],
    data: HillstromData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    random_seed: int,
) -> dict[str, object]:
    treatment = np.asarray(data.treatment)
    outcomes = np.asarray(data.conversion)
    treated_rows = [data.rows[index] for index in train_indices if treatment[index] == 1]
    control_rows = [data.rows[index] for index in train_indices if treatment[index] == 0]
    treated_model = factory(random_seed)
    control_model = factory(random_seed)
    started = time.perf_counter()
    treated_model.fit(treated_rows)
    control_model.fit(control_rows)
    fit_seconds = time.perf_counter() - started
    test_snapshots = [data.snapshots[index] for index in test_indices]
    started = time.perf_counter()
    treated_scores = np.asarray([score.score for score in treated_model.score(test_snapshots)])
    control_scores = np.asarray([score.score for score in control_model.score(test_snapshots)])
    score_seconds = time.perf_counter() - started
    uplift = treated_scores - control_scores
    top_count = max(1, int(len(test_indices) * 0.2))
    top_indices = test_indices[np.argsort(uplift)[::-1][:top_count]]
    top_treated_rate = _observed_rate(outcomes, treatment, top_indices, True)
    top_control_rate = _observed_rate(outcomes, treatment, top_indices, False)
    treatment_probability = float(treatment[train_indices].mean())
    policy = uplift > 0
    test_treatment = treatment[test_indices]
    test_outcomes = outcomes[test_indices]
    policy_value = np.where(
        policy & (test_treatment == 1),
        test_outcomes / treatment_probability,
        np.where(
            (~policy) & (test_treatment == 0),
            test_outcomes / (1 - treatment_probability),
            0,
        ),
    ).mean()
    return {
        "modelId": model_id,
        "fitSeconds": fit_seconds,
        "scoreSeconds": score_seconds,
        "trainTreatedRows": len(treated_rows),
        "trainControlRows": len(control_rows),
        "testRows": len(test_indices),
        "top20ObservedTreatedConversion": top_treated_rate,
        "top20ObservedControlConversion": top_control_rate,
        "top20ObservedUplift": top_treated_rate - top_control_rate,
        "policyContactShare": float(policy.mean()),
        "ipwPolicyConversion": float(policy_value),
        "runtime": {
            "treated": treated_model.runtime_info(),
            "control": control_model.runtime_info(),
        },
    }


def run_hillstrom_benchmark(
    path: Path,
    report_path: Path,
    *,
    random_seed: int = 7,
    profile: BenchmarkProfile = LAPTOP_PROFILE,
) -> dict[str, object]:
    data = load_hillstrom(path)
    treatment = np.asarray(data.treatment)
    outcomes = np.asarray(data.conversion)
    all_indices = np.arange(len(data.rows))
    test_mask = np.asarray(
        [
            int(hashlib.sha256(row.account_id.encode()).hexdigest()[:8], 16) % 5 == 0
            for row in data.rows
        ],
        dtype=np.bool_,
    )
    train_indices = all_indices[~test_mask]
    test_indices = all_indices[test_mask][:4_000]
    always_contact = _observed_rate(
        outcomes,
        treatment,
        test_indices,
        True,
    )
    never_contact = _observed_rate(
        outcomes,
        treatment,
        test_indices,
        False,
    )
    model_reports = []
    factories: tuple[tuple[str, Callable[[int], Scorer]], ...] = (
        ("lightgbm-v1", LightGbmScorer),
        ("tabpfn-local-v2", lambda seed: TabPfnScorer(seed, config=profile.tabpfn)),
    )
    for model_id, factory in factories:
        try:
            model_reports.append(
                _uplift_model_report(
                    model_id,
                    factory,
                    data,
                    train_indices,
                    test_indices,
                    random_seed,
                )
            )
        except Exception as error:
            model_reports.append(
                {
                    "modelId": model_id,
                    "status": "failed",
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
    report: dict[str, object] = {
        "schemaVersion": 1,
        "dataset": "Hillstrom Email Marketing",
        "source": (
            "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html"
        ),
        "contentSha256": sha256_file(path),
        "customers": len(data.rows),
        "treated": int(treatment.sum()),
        "control": int((treatment == 0).sum()),
        "conversions": int(outcomes.sum()),
        "conversionRate": float(outcomes.mean()),
        "averageTreatmentEffect": always_contact - never_contact,
        "testAlwaysContactConversion": always_contact,
        "testNeverContactConversion": never_contact,
        "target": "incremental conversion from email versus no email",
        "featureBoundary": (
            "Only pre-campaign customer history is used. Segment is the randomized "
            "treatment; visit, conversion, and spend are outcomes."
        ),
        "models": model_reports,
        "generatedAt": datetime.now(UTC).isoformat(),
        "executionProfile": profile.describe(),
        "hardware": hardware_summary(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


@dataclass(frozen=True)
class OutcomeDataset:
    key: str
    raw_filename: str
    report_stem: str
    runner: Callable[..., dict[str, object]]


OUTCOME_DATASETS: dict[str, OutcomeDataset] = {
    "olist": OutcomeDataset(
        "olist", "olist-marketing-funnel.zip", "olist-outcome-benchmark", run_olist_benchmark
    ),
    "uci-bank": OutcomeDataset(
        "uci-bank", "uci-bank-marketing.zip", "uci-bank-outcome-benchmark", run_uci_bank_benchmark
    ),
    "hillstrom": OutcomeDataset(
        "hillstrom", "hillstrom-email.csv.gz", "hillstrom-uplift-benchmark", run_hillstrom_benchmark
    ),
}


def summarize_report(report: dict[str, object]) -> list[dict[str, object]]:
    """Flatten a benchmark report into one row per model for tables and cross-profile diffs."""
    rows: list[dict[str, object]] = []
    runtimes = report.get("modelRuntimes")
    runtime_map = runtimes if isinstance(runtimes, dict) else {}
    benchmarks = report.get("benchmarks")
    if isinstance(benchmarks, list):
        for entry in benchmarks:
            runtime = runtime_map.get(entry["model_id"], {})
            rows.append(
                {
                    "modelId": entry["model_id"],
                    "status": entry["status"],
                    "prAuc": entry.get("pr_auc"),
                    "rocAuc": entry.get("roc_auc"),
                    "precisionAtK": entry.get("precision_at_k"),
                    "brier": entry.get("brier_score"),
                    "trainRows": entry.get("train_rows"),
                    "trainRowsUsed": runtime.get("trainingRowsUsed", entry.get("train_rows")),
                    "fitSeconds": entry.get("fit_seconds"),
                    "scoreSeconds": entry.get("score_seconds"),
                    "detail": entry.get("detail"),
                }
            )
    models = report.get("models")
    if isinstance(models, list):
        for entry in models:
            runtime = entry.get("runtime", {}).get("treated", {}) if "runtime" in entry else {}
            rows.append(
                {
                    "modelId": entry["modelId"],
                    "status": entry.get("status", "ok"),
                    "top20ObservedUplift": entry.get("top20ObservedUplift"),
                    "ipwPolicyConversion": entry.get("ipwPolicyConversion"),
                    "policyContactShare": entry.get("policyContactShare"),
                    "trainRows": entry.get("trainTreatedRows"),
                    "trainRowsUsed": runtime.get("trainingRowsUsed"),
                    "fitSeconds": entry.get("fitSeconds"),
                    "scoreSeconds": entry.get("scoreSeconds"),
                    "detail": entry.get("detail"),
                }
            )
    return rows


def run_outcome_benchmarks(
    root: Path,
    out_dir: Path,
    *,
    profile: BenchmarkProfile = LAPTOP_PROFILE,
    datasets: Sequence[str] = ("olist", "uci-bank", "hillstrom"),
    random_seed: int = 7,
    report_suffix: str = "v1",
    log: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run the real-outcome benchmarks under one execution profile and write a summary."""
    unknown = sorted(set(datasets) - set(OUTCOME_DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; choose from {sorted(OUTCOME_DATASETS)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schemaVersion": 1,
        "startedAt": datetime.now(UTC).isoformat(),
        "profile": profile.describe(),
        "hardware": hardware_summary(),
        "randomSeed": random_seed,
        "datasets": {},
    }
    dataset_summaries: dict[str, object] = {}
    for key in datasets:
        dataset = OUTCOME_DATASETS[key]
        raw_path = root / "data" / "raw" / dataset.raw_filename
        report_path = out_dir / f"{dataset.report_stem}-{report_suffix}.json"
        if log:
            log(f"[benchmark] {key}: {raw_path.name} -> {report_path}")
        started = time.perf_counter()
        if not raw_path.exists():
            dataset_summaries[key] = {"status": "missing_input", "rawPath": raw_path.as_posix()}
            continue
        report = dataset.runner(raw_path, report_path, random_seed=random_seed, profile=profile)
        rows = summarize_report(report)
        dataset_summaries[key] = {
            "status": "ok",
            "reportPath": report_path.as_posix(),
            "reportSha256": sha256_file(report_path),
            "wallSeconds": time.perf_counter() - started,
            "models": rows,
        }
        if log:
            for row in rows:
                log(f"[benchmark] {key}: {_format_summary_row(row)}")
    summary["datasets"] = dataset_summaries
    summary["finishedAt"] = datetime.now(UTC).isoformat()
    summary_path = out_dir / f"outcome-benchmark-summary-{report_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summaryPath"] = summary_path.as_posix()
    return summary


def _format_summary_row(row: dict[str, object]) -> str:
    def pct(value: object) -> str:
        return f"{float(value) * 100:5.1f}%" if isinstance(value, int | float) else "  n/a "

    parts = [f"{row['modelId']:<16}", f"status={row['status']}"]
    if "rocAuc" in row:
        parts.append(f"PR-AUC={pct(row['prAuc'])}")
        parts.append(f"ROC-AUC={pct(row['rocAuc'])}")
        parts.append(f"P@K={pct(row['precisionAtK'])}")
    if "top20ObservedUplift" in row:
        uplift = row["top20ObservedUplift"]
        parts.append(
            "top20 uplift="
            + (f"{float(uplift) * 100:+.3f}pp" if isinstance(uplift, int | float) else "n/a")
        )
        parts.append(f"IPW={pct(row['ipwPolicyConversion'])}")
    parts.append(f"rows_used={row.get('trainRowsUsed')}")
    fit = row.get("fitSeconds")
    score = row.get("scoreSeconds")
    if isinstance(fit, int | float) and isinstance(score, int | float):
        parts.append(f"fit={fit:.1f}s score={score:.1f}s")
    if row.get("detail"):
        parts.append(f"detail={row['detail']}")
    return " ".join(parts)
