"""Ablation harness: model baselines, split protocols, feature families, controls, layers.

Three ablation surfaces share one evaluation core (fit -> score -> metrics -> bootstrap):

- ``run_outcome_ablation``      real-outcome datasets (Olist, UCI Bank): model x protocol x
                                feature-family mask x negative control x seed;
- ``run_uplift_ablation``       Hillstrom: meta-learner x base learner x mask x seed with
                                Qini / AUUC / uplift@k / IPW policy value;
- ``run_architecture_ablation`` the manufacturing corpus: one feature layer at a time
                                (B1 raw latest values ... B7 StoryCard tags) plus controls,
                                with labels held fixed from the pristine corpus.

Every cell records what it actually ran (rows, prevalence, runtime evidence) so a number in a
report can be traced to a configuration, and nothing here writes to DuckDB.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import DecisionContext, load_decision_context
from .contracts import AccountRecord, AccountSnapshot, ModelScore, NormalizedEvent, TrainingRow
from .models import (
    LightGbmScorer,
    Scorer,
    SplitScores,
    TabPfnScorer,
    _matrix,
    classification_metrics,
    hardware_summary,
    point_in_time_split,
    score_split,
)
from .outcome_benchmarks import (
    LAPTOP_PROFILE,
    BenchmarkProfile,
    load_hillstrom,
    load_olist,
    load_uci_bank_marketing,
)
from .signal_registry import NumericFeature, SignalRegistry
from .store import ExperimentStore
from .training import build_training_materialization
from .uplift import UPLIFT_LEARNERS, base_learners, bootstrap_uplift, uplift_metrics

SnapshotIndex = dict[tuple[str, str], AccountSnapshot]
Log = Callable[[str], None] | None

# Scorers --------------------------------------------------------------------------------


class SklearnScorer(Scorer):
    """Adapter that lets any sklearn-style classifier act as an engine scorer."""

    def __init__(self, model_id: str, factory: Callable[[], Any], *, fill_nan: bool = False):
        self.model_id = model_id
        self.factory = factory
        self.fill_nan = fill_nan
        self.columns: tuple[str, ...] = ()
        self.model: Any = None
        self.training_rows_used = 0

    def _prepare(self, matrix: np.ndarray) -> np.ndarray:
        return np.nan_to_num(matrix, nan=0.0) if self.fill_nan else matrix

    def fit(self, rows: list[TrainingRow]) -> None:
        if len({row.label for row in rows}) < 2:
            raise ValueError(f"{self.model_id} requires both positive and negative labels")
        matrix, self.columns = _matrix([row.features for row in rows])
        try:
            self.model = self.factory()
        except ImportError as error:
            raise RuntimeError(
                f"{self.model_id} needs an optional dependency: install the `baselines` extra"
            ) from error
        self.model.fit(self._prepare(matrix), np.asarray([row.label for row in rows]))
        self.training_rows_used = len(rows)

    def score(self, snapshots: list[AccountSnapshot]) -> list[ModelScore]:
        if self.model is None:
            raise RuntimeError(f"{self.model_id} is not fitted")
        matrix, _ = _matrix([snapshot.features for snapshot in snapshots], self.columns)
        probabilities = np.asarray(self.model.predict_proba(self._prepare(matrix)))[:, 1]
        return [
            ModelScore(
                model_id=self.model_id,
                account_id=snapshot.account_id,
                score=float(np.clip(probability, 0.0, 1.0)),
                calibrated=False,
            )
            for snapshot, probability in zip(snapshots, probabilities, strict=True)
        ]

    def runtime_info(self) -> dict[str, object]:
        return {
            "estimator": type(self.model).__name__ if self.model is not None else None,
            "trainingRowsUsed": self.training_rows_used,
        }


class PositiveRateScorer(Scorer):
    """Predicts the training prevalence with a seeded tie-break, i.e. random targeting."""

    model_id = "positive-rate"

    def __init__(self, random_seed: int = 7):
        self.random_seed = random_seed
        self.rate = 0.0

    def fit(self, rows: list[TrainingRow]) -> None:
        self.rate = float(np.mean([row.label for row in rows])) if rows else 0.0

    def score(self, snapshots: list[AccountSnapshot]) -> list[ModelScore]:
        jitter = np.random.default_rng(self.random_seed).random(len(snapshots)) * 1e-6
        return [
            ModelScore(
                model_id=self.model_id,
                account_id=snapshot.account_id,
                score=float(np.clip(self.rate + noise - 5e-7, 0.0, 1.0)),
                calibrated=True,
            )
            for snapshot, noise in zip(snapshots, jitter, strict=True)
        ]

    def runtime_info(self) -> dict[str, object]:
        return {"trainingPrevalence": self.rate}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    description: str
    build: Callable[[int, BenchmarkProfile], Scorer]


def _lightgbm_factory(seed: int, **overrides: Any) -> Callable[[], Any]:
    def build() -> Any:
        from lightgbm import LGBMClassifier

        params: dict[str, Any] = {
            "n_estimators": 250,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "random_state": seed,
            "verbosity": -1,
        }
        params.update(overrides)
        return LGBMClassifier(**params)

    return build


def _xgboost_factory(seed: int) -> Callable[[], Any]:
    def build() -> Any:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )

    return build


def _catboost_factory(seed: int) -> Callable[[], Any]:
    def build() -> Any:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=seed,
            verbose=False,
            auto_class_weights="Balanced",
        )

    return build


def _logistic_factory(seed: int) -> Callable[[], Any]:
    def build() -> Any:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        )

    return build


def _random_forest_factory(seed: int) -> Callable[[], Any]:
    def build() -> Any:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    return build


MODEL_SPECS: dict[str, ModelSpec] = {
    "positive-rate": ModelSpec(
        "positive-rate",
        "Training prevalence with random tie-break (random targeting).",
        lambda seed, profile: PositiveRateScorer(seed),
    ),
    "logistic": ModelSpec(
        "logistic",
        "Standardized, class-balanced logistic regression.",
        lambda seed, profile: SklearnScorer("logistic-v1", _logistic_factory(seed), fill_nan=True),
    ),
    "random-forest": ModelSpec(
        "random-forest",
        "400-tree balanced random forest.",
        lambda seed, profile: SklearnScorer(
            "random-forest-v1", _random_forest_factory(seed), fill_nan=True
        ),
    ),
    "xgboost": ModelSpec(
        "xgboost",
        "XGBoost, 400 rounds, depth 5 (needs the `baselines` extra).",
        lambda seed, profile: SklearnScorer("xgboost-v1", _xgboost_factory(seed)),
    ),
    "catboost": ModelSpec(
        "catboost",
        "CatBoost, 500 iterations, balanced (needs the `baselines` extra).",
        lambda seed, profile: SklearnScorer("catboost-v1", _catboost_factory(seed)),
    ),
    "lightgbm-default": ModelSpec(
        "lightgbm-default",
        "LightGBM library defaults.",
        lambda seed, profile: SklearnScorer(
            "lightgbm-default-v1",
            _lightgbm_factory(
                seed,
                n_estimators=100,
                learning_rate=0.1,
                subsample=1.0,
                subsample_freq=0,
                colsample_bytree=1.0,
            ),
        ),
    ),
    "lightgbm": ModelSpec(
        "lightgbm",
        "Engine LightGBM: balanced, 250 trees, Platt fitted on expanding-window out-of-fold "
        "predictions, non-positive slopes rejected, final fit on every row.",
        lambda seed, profile: LightGbmScorer(seed),
    ),
    "lightgbm-holdout": ModelSpec(
        "lightgbm-holdout",
        "Previous engine LightGBM: oldest 80% trained, newest 20% held out for Platt (the "
        "configuration that inverted on UCI).",
        lambda seed, profile: LightGbmScorer(
            seed, calibration="holdout", model_id="lightgbm-holdout-v1"
        ),
    ),
    "lightgbm-full-train": ModelSpec(
        "lightgbm-full-train",
        "Engine LightGBM trained on every training row, raw probabilities, no calibration.",
        lambda seed, profile: LightGbmScorer(
            seed, calibration="none", model_id="lightgbm-full-train-v1"
        ),
    ),
    "lightgbm-regularized": ModelSpec(
        "lightgbm-regularized",
        "Shallow, regularized LightGBM (7 leaves, 400 trees, lr 0.02, lambda 5).",
        lambda seed, profile: SklearnScorer(
            "lightgbm-regularized-v1",
            _lightgbm_factory(
                seed,
                n_estimators=400,
                learning_rate=0.02,
                num_leaves=7,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
            ),
        ),
    ),
    "tabpfn": ModelSpec(
        "tabpfn",
        "TabPFN under the active execution profile (laptop 600 rows / GPU full context).",
        lambda seed, profile: TabPfnScorer(seed, config=profile.tabpfn),
    ),
}
DEFAULT_MODELS: tuple[str, ...] = tuple(MODEL_SPECS)


# Split protocols ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolSpec:
    key: str
    description: str
    split: Callable[[list[TrainingRow], int], tuple[list[TrainingRow], list[TrainingRow]]]


def _split_point_in_time(
    rows: list[TrainingRow], seed: int
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    del seed
    split = point_in_time_split(rows)
    return split.train, split.test


def _split_point_in_time_recent(
    rows: list[TrainingRow], seed: int
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    del seed
    split = point_in_time_split(rows)
    ordered = sorted(split.train, key=lambda row: (row.as_of, row.account_id))
    return ordered[int(len(ordered) * 0.67) :], split.test


def _stratified_holdout(
    rows: list[TrainingRow], seed: int, test_size: float = 0.2
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = np.asarray([row.label for row in rows])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(rows)), labels))
    return [rows[i] for i in train_idx], [rows[i] for i in test_idx]


def _split_random_late_period(
    rows: list[TrainingRow], seed: int
) -> tuple[list[TrainingRow], list[TrainingRow]]:
    split = point_in_time_split(rows)
    if not split.test:
        return _stratified_holdout(rows, seed)
    cutoff = min(row.as_of for row in split.test)
    return _stratified_holdout([row for row in rows if row.as_of >= cutoff], seed)


PROTOCOLS: dict[str, ProtocolSpec] = {
    "point_in_time": ProtocolSpec(
        "point_in_time",
        "Engine protocol: later 33% of dates is test, restricted to a hashed 20% account holdout.",
        _split_point_in_time,
    ),
    "point_in_time_recent": ProtocolSpec(
        "point_in_time_recent",
        "Same test partition; train only on the most recent third of the training window.",
        _split_point_in_time_recent,
    ),
    "random_stratified": ProtocolSpec(
        "random_stratified",
        "Seeded stratified 80/20 holdout over all rows (what most public numbers use).",
        _stratified_holdout,
    ),
    "random_late_period": ProtocolSpec(
        "random_late_period",
        "Stratified 80/20 holdout within the test era only (isolates drift from era).",
        _split_random_late_period,
    ),
}


# Feature families, masks, controls ------------------------------------------------------------

Predicate = Callable[[str], bool]


def _contains_any(*needles: str) -> Predicate:
    return lambda name: any(needle in name for needle in needles)


def _starts(prefix: str) -> Predicate:
    return lambda name: name.startswith(prefix)


FEATURE_FAMILIES: dict[str, dict[str, Predicate]] = {
    "uci-bank": {
        "macro": _contains_any(
            "euribor3m", "emp.var.rate", "nr.employed", "cons.price.idx", "cons.conf.idx"
        ),
        "calendar": _contains_any(".month.", ".day_of_week."),
        "prior_contact": _contains_any("pdays", "previous", "poutcome", ".campaign"),
        "client": _contains_any(
            ".age", ".job.", ".marital.", ".education.", ".default.", ".housing.", ".loan."
        ),
        "channel": _starts("contact.contact."),
    },
    "olist": {
        "origin": _starts("origin."),
        "landing_page": _starts("landing_page."),
        "calendar": _starts("contact."),
    },
    "hillstrom": {
        "recency": lambda name: name == "customer.recency",
        "history": _starts("customer.history"),
        "category": lambda name: name in {"customer.mens", "customer.womens"},
        "newbie": lambda name: name == "customer.newbie",
        "zip": _starts("customer.zip_code."),
        "channel": _starts("customer.channel."),
    },
}


@dataclass(frozen=True)
class MaskSpec:
    """``none`` | ``drop:a+b`` | ``only:a+b`` over named feature families."""

    key: str
    mode: str
    families: tuple[str, ...]

    @classmethod
    def parse(cls, text: str) -> MaskSpec:
        text = text.strip()
        if text == "none":
            return cls("none", "none", ())
        if ":" not in text:
            raise ValueError(f"mask {text!r} must be 'none', 'drop:<family>' or 'only:<family>'")
        mode, families = text.split(":", 1)
        if mode not in {"drop", "only"}:
            raise ValueError(f"mask mode {mode!r} must be 'drop' or 'only'")
        return cls(text, mode, tuple(part for part in families.split("+") if part))

    def keep(self, families: Mapping[str, Predicate]) -> Predicate:
        unknown = sorted(set(self.families) - set(families))
        if unknown:
            raise ValueError(f"unknown feature families {unknown}; known: {sorted(families)}")
        predicates = [families[name] for name in self.families]
        if self.mode == "none":
            return lambda name: True
        if self.mode == "drop":
            return lambda name: not any(predicate(name) for predicate in predicates)
        return lambda name: name == "fit.score" or any(predicate(name) for predicate in predicates)


def apply_feature_mask(
    rows: list[TrainingRow], snapshots: SnapshotIndex, keep: Predicate
) -> tuple[list[TrainingRow], SnapshotIndex, int]:
    """Drop features by name from rows and their snapshots; returns the surviving column count."""
    names = {name for row in rows for name in row.features}
    kept = {name for name in names if keep(name)}
    if kept == names:
        return rows, snapshots, len(names)
    new_rows = [
        row.model_copy(update={"features": {k: v for k, v in row.features.items() if k in kept}})
        for row in rows
    ]
    new_snapshots = {
        key: snapshot.model_copy(
            update={"features": {k: v for k, v in snapshot.features.items() if k in kept}}
        )
        for key, snapshot in snapshots.items()
    }
    return new_rows, new_snapshots, len(kept)


PRE_SPLIT_CONTROLS = ("shuffled_as_of",)
POST_SPLIT_CONTROLS = ("shuffled_labels",)
CONTROLS: dict[str, str] = {
    "none": "No manipulation.",
    "shuffled_labels": "Permute training labels; every model must fall to chance.",
    "shuffled_as_of": (
        "Permute row timestamps before splitting; destroys chronology in time-based protocols."
    ),
}


def apply_pre_split_control(
    rows: list[TrainingRow], snapshots: SnapshotIndex, control: str, seed: int
) -> tuple[list[TrainingRow], SnapshotIndex]:
    if control != "shuffled_as_of":
        return rows, snapshots
    rng = np.random.default_rng(seed)
    permuted = [rows[i].as_of for i in rng.permutation(len(rows))]
    new_rows: list[TrainingRow] = []
    new_snapshots: SnapshotIndex = {}
    for row, as_of in zip(rows, permuted, strict=True):
        snapshot = snapshots[(row.account_id, row.as_of.isoformat())]
        new_row = row.model_copy(update={"as_of": as_of})
        new_rows.append(new_row)
        new_snapshots[(row.account_id, as_of.isoformat())] = snapshot.model_copy(
            update={"as_of": as_of}
        )
    return new_rows, new_snapshots


def apply_post_split_control(
    train: list[TrainingRow], control: str, seed: int
) -> list[TrainingRow]:
    if control != "shuffled_labels":
        return train
    labels = np.asarray([row.label for row in train])
    permuted = labels[np.random.default_rng(seed).permutation(len(labels))]
    return [
        row.model_copy(update={"label": int(label)})
        for row, label in zip(train, permuted, strict=True)
    ]


# Bootstrap + shared evaluation ------------------------------------------------------------------


def bootstrap_classification(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    samples: int,
    seed: int,
    precision_k: int,
) -> dict[str, list[float]]:
    """Percentile 95% intervals over test-row resamples for ROC-AUC, PR-AUC, precision@K."""
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"rocAuc": [], "prAuc": [], "precisionAtK": []}
    n = len(labels)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        sample_labels = labels[idx]
        if sample_labels.min() == sample_labels.max():
            continue
        metrics = classification_metrics(sample_labels, probabilities[idx], precision_k)
        for key, metric in (
            ("rocAuc", "roc_auc"),
            ("prAuc", "pr_auc"),
            ("precisionAtK", "precision_at_k"),
        ):
            value = metrics[metric]
            if value is not None:
                draws[key].append(value)
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in draws.items()
        if values
    }


def _seeds(count: int, base: int) -> tuple[int, ...]:
    return tuple(base + offset for offset in range(max(1, count)))


def evaluate_cell(
    scorer: Scorer,
    train: list[TrainingRow],
    test: list[TrainingRow],
    snapshots: SnapshotIndex,
    *,
    precision_k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Fit/score one configuration and return a flat, JSON-ready cell."""
    cell: dict[str, object] = {
        "model": scorer.model_id,
        "seed": seed,
        "trainRows": len(train),
        "testRows": len(test),
        "trainPrevalence": float(np.mean([row.label for row in train])) if train else None,
        "testPrevalence": float(np.mean([row.label for row in test])) if test else None,
        "featureCount": len({name for row in train for name in row.features}),
        "trainPositives": int(sum(row.label for row in train)),
        "testPositives": int(sum(row.label for row in test)),
    }
    if not train or not test:
        cell.update({"status": "failed", "detail": "empty partition"})
        return cell
    try:
        scored: SplitScores = score_split(scorer, train, test, snapshots)
    except RuntimeError as error:
        cell.update({"status": "unavailable", "detail": str(error)})
        return cell
    except Exception as error:
        cell.update({"status": "failed", "detail": f"{type(error).__name__}: {error}"})
        return cell
    metrics = classification_metrics(scored.labels, scored.probabilities, precision_k)
    degenerate = metrics["roc_auc"] is None
    cell.update(
        {
            # A single-class test partition is a sample-size problem, not a model result.
            "status": "degenerate" if degenerate else "ok",
            "detail": "test partition has a single class" if degenerate else None,
            "rocAuc": metrics["roc_auc"],
            "prAuc": metrics["pr_auc"],
            "precisionAtK": metrics["precision_at_k"],
            "brier": metrics["brier_score"],
            "precisionK": precision_k,
            "fitSeconds": scored.fit_seconds,
            "scoreSeconds": scored.score_seconds,
            "ci95": bootstrap_classification(
                scored.labels,
                scored.probabilities,
                samples=bootstrap_samples,
                seed=seed,
                precision_k=precision_k,
            ),
            "runtime": scorer.runtime_info(),
        }
    )
    return cell


def aggregate_cells(
    cells: Sequence[dict[str, object]], group_keys: Sequence[str]
) -> list[dict[str, object]]:
    """Mean and standard deviation across seeds for every configuration group."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for cell in cells:
        groups.setdefault(tuple(cell.get(key) for key in group_keys), []).append(cell)
    rows: list[dict[str, object]] = []
    for key, members in groups.items():
        row: dict[str, object] = dict(zip(group_keys, key, strict=True))
        ok = [cell for cell in members if cell.get("status") == "ok"]
        row["seeds"] = len(members)
        row["ok"] = len(ok)
        row["status"] = "ok" if ok else str(members[0].get("status"))
        if not ok:
            row["detail"] = members[0].get("detail")
        for metric in (
            "rocAuc",
            "prAuc",
            "precisionAtK",
            "brier",
            "qiniNormalized",
            "upliftAt20",
            "ipwPolicyTop20",
            "fitSeconds",
        ):
            values = [
                float(value) for cell in ok if isinstance((value := cell.get(metric)), int | float)
            ]
            if values:
                row[f"{metric}Mean"] = float(np.mean(values))
                row[f"{metric}Std"] = float(np.std(values)) if len(values) > 1 else 0.0
        first_ci = next((cell.get("ci95") for cell in ok if cell.get("ci95")), None)
        if isinstance(first_ci, dict):
            row["ci95FirstSeed"] = first_ci
        rows.append(row)
    return rows


def _format_pct(value: object) -> str:
    return f"{float(value) * 100:5.1f}%" if isinstance(value, int | float) else "  n/a "


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, int | float) else float("nan")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


# Outcome datasets -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeAblationDataset:
    key: str
    raw_filename: str
    precision_k: int
    load: Callable[[Path], tuple[list[TrainingRow], SnapshotIndex]]


def _load_olist(path: Path) -> tuple[list[TrainingRow], SnapshotIndex]:
    data = load_olist(path)
    return data.rows, data.snapshots_by_key


def _load_uci(path: Path) -> tuple[list[TrainingRow], SnapshotIndex]:
    data = load_uci_bank_marketing(path)
    return data.rows, data.snapshots_by_key


OUTCOME_ABLATION_DATASETS: dict[str, OutcomeAblationDataset] = {
    "olist": OutcomeAblationDataset("olist", "olist-marketing-funnel.zip", 100, _load_olist),
    "uci-bank": OutcomeAblationDataset("uci-bank", "uci-bank-marketing.zip", 250, _load_uci),
}


def _resolve(keys: Sequence[str], registry: Mapping[str, Any], what: str) -> list[str]:
    unknown = sorted(set(keys) - set(registry))
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; choose from {sorted(registry)}")
    return list(keys)


def run_outcome_ablation(
    root: Path,
    out_dir: Path,
    *,
    datasets: Sequence[str] = ("uci-bank", "olist"),
    models: Sequence[str] = DEFAULT_MODELS,
    protocols: Sequence[str] = ("point_in_time", "random_stratified"),
    masks: Sequence[str] = ("none",),
    controls: Sequence[str] = ("none",),
    seeds: int = 1,
    base_seed: int = 7,
    bootstrap_samples: int = 200,
    profile: BenchmarkProfile = LAPTOP_PROFILE,
    report_suffix: str = "v1",
    log: Log = None,
) -> dict[str, object]:
    dataset_keys = _resolve(datasets, OUTCOME_ABLATION_DATASETS, "datasets")
    model_keys = _resolve(models, MODEL_SPECS, "models")
    protocol_keys = _resolve(protocols, PROTOCOLS, "protocols")
    control_keys = _resolve(controls, CONTROLS, "controls")
    mask_specs = [MaskSpec.parse(mask) for mask in masks]
    started = datetime.now(UTC)
    reports: dict[str, object] = {}
    for dataset_key in dataset_keys:
        dataset = OUTCOME_ABLATION_DATASETS[dataset_key]
        raw_path = root / "data" / "raw" / dataset.raw_filename
        if not raw_path.exists():
            reports[dataset_key] = {"status": "missing_input", "rawPath": raw_path.as_posix()}
            continue
        rows, snapshots = dataset.load(raw_path)
        families = FEATURE_FAMILIES[dataset_key]
        cells: list[dict[str, object]] = []
        for mask in mask_specs:
            masked_rows, masked_snapshots, feature_count = apply_feature_mask(
                rows, snapshots, mask.keep(families)
            )
            for control in control_keys:
                for protocol_key in protocol_keys:
                    protocol = PROTOCOLS[protocol_key]
                    for seed in _seeds(seeds, base_seed):
                        pre_rows, pre_snapshots = apply_pre_split_control(
                            masked_rows, masked_snapshots, control, seed
                        )
                        train, test = protocol.split(pre_rows, seed)
                        train = apply_post_split_control(train, control, seed)
                        for model_key in model_keys:
                            scorer = MODEL_SPECS[model_key].build(seed, profile)
                            cell = evaluate_cell(
                                scorer,
                                train,
                                test,
                                pre_snapshots,
                                precision_k=dataset.precision_k,
                                bootstrap_samples=bootstrap_samples,
                                seed=seed,
                            )
                            cell.update(
                                {
                                    "dataset": dataset_key,
                                    "modelKey": model_key,
                                    "protocol": protocol_key,
                                    "mask": mask.key,
                                    "control": control,
                                    "maskedFeatureCount": feature_count,
                                }
                            )
                            cells.append(cell)
                            if log:
                                log(_format_cell(cell))
        aggregates = aggregate_cells(cells, ("dataset", "protocol", "mask", "control", "modelKey"))
        report = {
            "schemaVersion": 1,
            "kind": "outcome-ablation",
            "dataset": dataset_key,
            "generatedAt": datetime.now(UTC).isoformat(),
            "profile": profile.describe(),
            "hardware": hardware_summary(),
            "grid": {
                "models": model_keys,
                "protocols": protocol_keys,
                "masks": [mask.key for mask in mask_specs],
                "controls": control_keys,
                "seeds": list(_seeds(seeds, base_seed)),
                "bootstrapSamples": bootstrap_samples,
                "precisionK": dataset.precision_k,
            },
            "cells": cells,
            "aggregates": aggregates,
        }
        report_path = out_dir / f"{dataset_key}-ablation-{report_suffix}.json"
        _write_report(report_path, report)
        reports[dataset_key] = {
            "status": "ok",
            "reportPath": report_path.as_posix(),
            "cells": len(cells),
        }
    summary = {
        "schemaVersion": 1,
        "kind": "outcome-ablation-summary",
        "startedAt": started.isoformat(),
        "finishedAt": datetime.now(UTC).isoformat(),
        "profile": profile.describe(),
        "datasets": reports,
    }
    _write_report(out_dir / f"outcome-ablation-summary-{report_suffix}.json", summary)
    return summary


def _format_cell(cell: dict[str, object]) -> str:
    parts = [
        f"[ablate] {cell.get('dataset') or cell.get('arm')}",
        f"{cell.get('protocol', '')}",
        f"mask={cell.get('mask', 'none')}",
        f"ctrl={cell.get('control', 'none')}",
        f"seed={cell.get('seed')}",
        f"{cell.get('modelKey') or cell.get('model')!s:<20}",
        f"status={cell.get('status')}",
    ]
    if cell.get("status") == "ok":
        if "rocAuc" in cell:
            parts.append(
                f"ROC={_format_pct(cell.get('rocAuc'))} PR={_format_pct(cell.get('prAuc'))} "
                f"P@K={_format_pct(cell.get('precisionAtK'))}"
            )
            ci = cell.get("ci95")
            if isinstance(ci, dict) and "rocAuc" in ci:
                low, high = ci["rocAuc"]
                parts.append(f"ROC95=[{low * 100:.1f},{high * 100:.1f}]")
        if "qiniNormalized" in cell:
            qini = _as_float(cell.get("qiniNormalized"))
            uplift = _as_float(cell.get("upliftAt20"))
            parts.append(f"qini={qini:.4f} uplift@20={uplift * 100:+.3f}pp")
        parts.append(
            f"rows={cell.get('trainRows')}/{cell.get('testRows')} "
            f"pos={cell.get('trainPositives')}/{cell.get('testPositives')} "
            f"feats={cell.get('featureCount')}"
        )
    else:
        parts.append(str(cell.get("detail"))[:120])
    return " ".join(parts)


# Hillstrom uplift ---------------------------------------------------------------------------------

DEFAULT_UPLIFT_LEARNERS: tuple[str, ...] = tuple(UPLIFT_LEARNERS)
DEFAULT_BASE_LEARNERS: tuple[str, ...] = ("lightgbm", "logistic", "tabpfn")


def _hillstrom_split(
    account_ids: Sequence[str],
    treatment: np.ndarray,
    outcomes: np.ndarray,
    protocol: str,
    seed: int,
    test_cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    import hashlib

    all_indices = np.arange(len(account_ids))
    cap = test_cap if test_cap > 0 else None
    if protocol == "hashed_holdout":
        mask = np.asarray(
            [int(hashlib.sha256(a.encode()).hexdigest()[:8], 16) % 5 == 0 for a in account_ids],
            dtype=bool,
        )
        return all_indices[~mask], all_indices[mask][:cap]
    if protocol == "random_stratified":
        from sklearn.model_selection import StratifiedShuffleSplit

        strata = treatment * 2 + outcomes
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(strata)), strata))
        return np.asarray(train_idx), np.asarray(test_idx)[:cap]
    raise ValueError(f"unknown uplift protocol {protocol!r}")


def run_uplift_ablation(
    root: Path,
    out_dir: Path,
    *,
    learners: Sequence[str] = DEFAULT_UPLIFT_LEARNERS,
    bases: Sequence[str] = DEFAULT_BASE_LEARNERS,
    masks: Sequence[str] = ("none",),
    protocols: Sequence[str] = ("hashed_holdout",),
    seeds: int = 1,
    base_seed: int = 7,
    bootstrap_samples: int = 200,
    test_cap: int = 0,
    profile: BenchmarkProfile = LAPTOP_PROFILE,
    report_suffix: str = "v1",
    log: Log = None,
) -> dict[str, object]:
    """Hillstrom meta-learner grid. ``test_cap`` 0 keeps the whole holdout (~12.8k rows);
    the original benchmark's 4,000-row cap leaves only ~35 conversions and is far too noisy
    for Qini/AUUC comparisons."""
    raw_path = root / "data" / "raw" / "hillstrom-email.csv.gz"
    data = load_hillstrom(raw_path)
    available_bases = base_learners(profile.tabpfn)
    learner_keys = _resolve(learners, UPLIFT_LEARNERS, "uplift learners")
    base_keys = _resolve(bases, available_bases, "base learners")
    mask_specs = [MaskSpec.parse(mask) for mask in masks]
    matrix, columns = _matrix([row.features for row in data.rows])
    treatment = np.asarray(data.treatment)
    outcomes = np.asarray(data.conversion)
    account_ids = [row.account_id for row in data.rows]
    families = FEATURE_FAMILIES["hillstrom"]
    cells: list[dict[str, object]] = []
    for mask in mask_specs:
        keep = mask.keep(families)
        kept_columns = [i for i, name in enumerate(columns) if keep(name)]
        X = matrix[:, kept_columns]
        for protocol in protocols:
            for seed in _seeds(seeds, base_seed):
                train_idx, test_idx = _hillstrom_split(
                    account_ids, treatment, outcomes, protocol, seed, test_cap
                )
                propensity = float(treatment[train_idx].mean())
                reference = uplift_metrics(
                    np.zeros(len(test_idx)), treatment[test_idx], outcomes[test_idx], propensity
                )
                for learner_key in learner_keys:
                    for base_key in base_keys:
                        if learner_key == "random" and base_key != base_keys[0]:
                            continue
                        cell: dict[str, object] = {
                            "dataset": "hillstrom",
                            "learner": learner_key,
                            "modelKey": f"{learner_key}/{base_key}",
                            "base": base_key,
                            "protocol": protocol,
                            "mask": mask.key,
                            "control": "none",
                            "seed": seed,
                            "trainRows": len(train_idx),
                            "testRows": len(test_idx),
                            "featureCount": len(kept_columns),
                            "propensity": propensity,
                            "testAte": reference["testAte"],
                            "ipwContactAll": reference["ipwContactAll"],
                            "ipwContactNone": reference["ipwContactNone"],
                        }
                        started = time.perf_counter()
                        try:
                            fitted = UPLIFT_LEARNERS[learner_key](
                                available_bases[base_key],
                                seed,
                                X[train_idx],
                                treatment[train_idx],
                                outcomes[train_idx],
                                X[test_idx],
                                propensity,
                            )
                        except RuntimeError as error:
                            cell.update({"status": "unavailable", "detail": str(error)})
                        except Exception as error:
                            cell.update(
                                {"status": "failed", "detail": f"{type(error).__name__}: {error}"}
                            )
                        else:
                            metrics = uplift_metrics(
                                fitted.scores, treatment[test_idx], outcomes[test_idx], propensity
                            )
                            cell.update(metrics)
                            cell.update(
                                {
                                    "status": "ok",
                                    "upliftAt20": metrics["upliftAt"]["20"]["uplift"],  # type: ignore[index]
                                    "fitSeconds": time.perf_counter() - started,
                                    "learnerDetail": fitted.detail,
                                    "ci95": bootstrap_uplift(
                                        fitted.scores,
                                        treatment[test_idx],
                                        outcomes[test_idx],
                                        propensity,
                                        samples=bootstrap_samples,
                                        seed=seed,
                                    ),
                                }
                            )
                        cells.append(cell)
                        if log:
                            log(_format_cell(cell))
    aggregates = aggregate_cells(cells, ("dataset", "protocol", "mask", "learner", "base"))
    report = {
        "schemaVersion": 1,
        "kind": "uplift-ablation",
        "dataset": "hillstrom",
        "generatedAt": datetime.now(UTC).isoformat(),
        "profile": profile.describe(),
        "hardware": hardware_summary(),
        "grid": {
            "learners": learner_keys,
            "bases": base_keys,
            "masks": [mask.key for mask in mask_specs],
            "protocols": list(protocols),
            "seeds": list(_seeds(seeds, base_seed)),
            "bootstrapSamples": bootstrap_samples,
            "testCap": test_cap,
        },
        "cells": cells,
        "aggregates": aggregates,
    }
    report_path = out_dir / f"hillstrom-uplift-ablation-{report_suffix}.json"
    _write_report(report_path, report)
    return {"status": "ok", "reportPath": report_path.as_posix(), "cells": len(cells)}


# Architecture layers (manufacturing corpus) ----------------------------------------------------

ARCHITECTURE_DATES: tuple[datetime, ...] = tuple(
    datetime(year, month, day, tzinfo=UTC)
    for year, month, day in (
        (2024, 6, 30),
        (2024, 12, 31),
        (2025, 3, 31),
        (2025, 6, 30),
        (2025, 9, 30),
        (2025, 12, 31),
    )
)
LABEL_SIGNAL_TYPES = {"sba_loan", "usaspending_award", "job_posting"}

_LIFECYCLE = re.compile(
    r"^signal\.[^.]+\.(event_count|age_days|days_since_last_change|latest_strength|maximum_strength|source_latency_days)\.(max|mean)$"
)
_STAGE = re.compile(
    r"^signal\.[^.]+\.(material_change_count|state_transition_count|is_terminal)\.(max|mean)$"
)
_LATEST = re.compile(r"^signal\.[^.]+\.[^.]+\.latest\.(max|mean)$")
_REDUCERS = re.compile(r"^signal\.[^.]+\.[^.]+\.(max|mean|sum|delta|slope)\.(max|mean)$")
_ACTIVE = re.compile(r"^signal\.[^.]+\.active_count$")
_DECAY = re.compile(r"^signal\.[^.]+\.(decayed_score|max_instance_score)$")
_ROLLUP = {"signals.active_count", "signals.distinct_types", "signals.corroboration"}

ARCHITECTURE_FAMILIES: dict[str, Predicate] = {
    "account": lambda n: n == "account.employee_count",
    "fit": _starts("fit."),
    "latest": lambda n: bool(_LATEST.match(n)),
    "lifecycle": lambda n: bool(_LIFECYCLE.match(n)),
    "stage": lambda n: bool(_STAGE.match(n)),
    "reducers": lambda n: bool(_REDUCERS.match(n)) and not _LATEST.match(n),
    "active_count": lambda n: bool(_ACTIVE.match(n)),
    "decay": lambda n: bool(_DECAY.match(n)) or n == "signals.weighted_sum",
    "rollup": lambda n: n in _ROLLUP,
    "sequence": _starts("sequence."),
    "story": _starts("story."),
    "sba": _contains_any(".sba_loan.", "financed_then"),
    "osha": _contains_any(".osha_injury_summary.", "safety_pressure"),
}


def _families(*names: str) -> Predicate:
    predicates = [ARCHITECTURE_FAMILIES[name] for name in names]
    return lambda feature: any(predicate(feature) for predicate in predicates)


def _all_but(*names: str) -> Predicate:
    predicates = [ARCHITECTURE_FAMILIES[name] for name in names]
    return lambda feature: not any(predicate(feature) for predicate in predicates)


@dataclass(frozen=True)
class ArchitectureArm:
    key: str
    description: str
    keep: Predicate
    variant: str = "full"
    context_fn: Callable[[DecisionContext], DecisionContext] | None = None
    registry_fn: Callable[[SignalRegistry], SignalRegistry] | None = None
    events_fn: Callable[[list[NormalizedEvent], int], list[NormalizedEvent]] | None = None
    story_features: bool = False
    post_split_control: str = "none"


def _context_no_sequences(context: DecisionContext) -> DecisionContext:
    return DecisionContext(
        context.product,
        context.icp,
        context.goal,
        context.signal_policy.model_copy(update={"sequence_rules": ()}),
    )


def _context_reversed_sequences(context: DecisionContext) -> DecisionContext:
    rules = tuple(
        rule.model_copy(update={"ordered_signal_types": tuple(reversed(rule.ordered_signal_types))})
        for rule in context.signal_policy.sequence_rules
    )
    return DecisionContext(
        context.product,
        context.icp,
        context.goal,
        context.signal_policy.model_copy(update={"sequence_rules": rules}),
    )


def _context_no_decay(context: DecisionContext) -> DecisionContext:
    entries = tuple(
        entry.model_copy(update={"half_life_days": 1e9}) for entry in context.signal_policy.entries
    )
    return DecisionContext(
        context.product,
        context.icp,
        context.goal,
        context.signal_policy.model_copy(update={"entries": entries}),
    )


def _registry_latest_only(registry: SignalRegistry) -> SignalRegistry:
    definitions = tuple(
        definition.model_copy(
            update={
                "numeric_features": tuple(
                    NumericFeature(field=feature.field, unit=feature.unit, reducers=("latest",))
                    for feature in definition.numeric_features
                )
            }
        )
        for definition in registry.all()
    )
    return SignalRegistry(definitions)


def _registry_no_stage(registry: SignalRegistry) -> SignalRegistry:
    definitions = tuple(
        definition.model_copy(
            update={"state_field": None, "terminal_states": (), "material_kinds": ()}
        )
        for definition in registry.all()
    )
    return SignalRegistry(definitions)


def shuffle_history_timestamps(events: list[NormalizedEvent], seed: int) -> list[NormalizedEvent]:
    """Permute timestamps among each account's events that were visible before the first as_of.

    Only pre-history events are touched, so no label-window event can leak into the features
    and the ``public_future_event`` labels are unchanged by construction.
    """
    cutoff = min(ARCHITECTURE_DATES)
    rng = np.random.default_rng(seed)
    by_account: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        if event.available_at <= cutoff:
            by_account.setdefault(event.account_id, []).append(index)
    shuffled = list(events)
    for indices in by_account.values():
        if len(indices) < 2:
            continue
        stamps = [
            (
                events[i].occurred_at,
                events[i].delivered_at,
                events[i].available_at,
                events[i].ingested_at,
            )
            for i in indices
        ]
        for target, source in zip(indices, rng.permutation(len(indices)), strict=True):
            occurred, delivered, available, ingested = stamps[source]
            shuffled[target] = events[target].model_copy(
                update={
                    "occurred_at": occurred,
                    "delivered_at": delivered,
                    "available_at": available,
                    "ingested_at": ingested,
                }
            )
    return shuffled


ARCHITECTURE_ARMS: dict[str, ArchitectureArm] = {
    "B1_raw_latest": ArchitectureArm(
        "B1_raw_latest",
        "Latest source values only plus account size.",
        _families("latest", "account"),
    ),
    "B2_pit_recency": ArchitectureArm(
        "B2_pit_recency",
        "B1 + point-in-time lifecycle/recency descriptors and instance counts.",
        _families("latest", "account", "lifecycle", "active_count"),
    ),
    "B3_factsheet": ArchitectureArm(
        "B3_factsheet",
        "B2 + full FactSheet reducers (max/mean/sum/delta/slope) and stage facts.",
        _families("latest", "account", "lifecycle", "active_count", "reducers", "stage"),
    ),
    "B4_decay": ArchitectureArm(
        "B4_decay",
        "B3 + stage-aware decayed scores and account roll-ups.",
        _families(
            "latest", "account", "lifecycle", "active_count", "reducers", "stage", "decay", "rollup"
        ),
    ),
    "B5_sequences": ArchitectureArm(
        "B5_sequences",
        "B4 + cross-signal sequence features.",
        _all_but("fit", "story"),
    ),
    "B6_icp": ArchitectureArm(
        "B6_icp", "B5 + ICP conditioning (current engine feature set).", _all_but("story")
    ),
    "B7_story_tags": ArchitectureArm(
        "B7_story_tags",
        "B6 + deterministic StoryCard phase tags.",
        lambda n: True,
        variant="story",
        story_features=True,
    ),
    "C_shuffled_labels": ArchitectureArm(
        "C_shuffled_labels",
        "B6 features, training labels permuted (must fall to chance).",
        _all_but("story"),
        post_split_control="shuffled_labels",
    ),
    "C_shuffled_timestamps": ArchitectureArm(
        "C_shuffled_timestamps",
        "B6 features on history with per-account timestamp permutation (labels unchanged).",
        _all_but("story"),
        variant="shuffled_timestamps",
        events_fn=shuffle_history_timestamps,
    ),
    "C_no_sequences": ArchitectureArm(
        "C_no_sequences",
        "B6 with sequence rules removed.",
        _all_but("story"),
        variant="no_sequences",
        context_fn=_context_no_sequences,
    ),
    "C_sequence_reversed": ArchitectureArm(
        "C_sequence_reversed",
        "B6 with every sequence rule's order reversed.",
        _all_but("story"),
        variant="reversed_sequences",
        context_fn=_context_reversed_sequences,
    ),
    "C_latest_only": ArchitectureArm(
        "C_latest_only",
        "B6 with numeric reducers restricted to `latest` (no max/mean/delta/slope).",
        _all_but("story"),
        variant="latest_only",
        registry_fn=_registry_latest_only,
    ),
    "C_no_decay": ArchitectureArm(
        "C_no_decay",
        "B6 with decay disabled (half-life 1e9 days).",
        _all_but("story"),
        variant="no_decay",
        context_fn=_context_no_decay,
    ),
    "C_no_stage": ArchitectureArm(
        "C_no_stage",
        "B6 with state/stage tracking removed from every signal definition.",
        _all_but("story"),
        variant="no_stage",
        registry_fn=_registry_no_stage,
    ),
    "C_no_sba": ArchitectureArm(
        "C_no_sba",
        "B6 without any SBA-derived feature (labels unchanged).",
        _all_but("story", "sba"),
    ),
    "C_no_osha": ArchitectureArm(
        "C_no_osha", "B6 without any OSHA-derived feature.", _all_but("story", "osha")
    ),
    "C_missingness_only": ArchitectureArm(
        "C_missingness_only",
        "Only which signal families are observed (instance counts, coverage roll-ups, size); "
        "if this matches B6 the proxy label is population identification, not timing.",
        _families("active_count", "rollup", "account"),
    ),
}
DEFAULT_ARCHITECTURE_ARMS: tuple[str, ...] = tuple(ARCHITECTURE_ARMS)
DEFAULT_ARCHITECTURE_MODELS: tuple[str, ...] = (
    "positive-rate",
    "logistic",
    "lightgbm",
    "lightgbm-full-train",
    "tabpfn",
)


@dataclass
class ManufacturingCorpus:
    accounts: list[AccountRecord]
    events: list[NormalizedEvent]
    source: str
    account_count: int = 0
    event_count: int = 0
    signal_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.account_count = len(self.accounts)
        self.event_count = len(self.events)
        self.signal_types = sorted({event.signal_type for event in self.events})


def load_manufacturing_corpus(path: Path, max_accounts: int) -> ManufacturingCorpus:
    """First-N accounts by id (the same deterministic sample runs use) and all their events.

    ``path`` is either the DuckDB store or a gzip JSON extract from
    ``export_manufacturing_corpus`` (small enough to ship to a remote GPU instance).
    """
    if path.suffix != ".duckdb":
        return _load_corpus_extract(path, max_accounts)
    store = ExperimentStore(path)
    try:
        ids = [
            str(row[0])
            for row in store.connection.execute(
                "SELECT account_id FROM account ORDER BY account_id LIMIT ?", [max_accounts]
            ).fetchall()
        ]
        by_id = store.accounts_by_ids(ids)
        accounts = [by_id[account_id] for account_id in ids if account_id in by_id]
        placeholders = ",".join("?" for _ in ids)
        bodies = store.connection.execute(
            f"SELECT body::VARCHAR FROM normalized_event WHERE account_id IN ({placeholders}) "
            "ORDER BY available_at, event_id",
            ids,
        ).fetchall()
    finally:
        store.close()
    events = [NormalizedEvent.model_validate_json(str(body[0])) for body in bodies]
    return ManufacturingCorpus(accounts, events, path.as_posix())


def export_manufacturing_corpus(
    db_path: Path, max_accounts: int, target: Path
) -> dict[str, object]:
    """Write the first-N accounts and their events as gzip JSON for remote execution."""
    import gzip

    corpus = load_manufacturing_corpus(db_path, max_accounts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "source": corpus.source,
                    "maxAccounts": max_accounts,
                    "exportedAt": datetime.now(UTC).isoformat(),
                    "accounts": [account.model_dump(mode="json") for account in corpus.accounts],
                    "events": [event.model_dump(mode="json") for event in corpus.events],
                }
            )
        )
    return {
        "path": target.as_posix(),
        "accounts": corpus.account_count,
        "events": corpus.event_count,
        "bytes": target.stat().st_size,
    }


def _load_corpus_extract(path: Path, max_accounts: int) -> ManufacturingCorpus:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    accounts = [AccountRecord.model_validate(item) for item in payload["accounts"]][:max_accounts]
    selected = {account.account_id for account in accounts}
    events = [
        event
        for event in (NormalizedEvent.model_validate(item) for item in payload["events"])
        if event.account_id in selected
    ]
    return ManufacturingCorpus(accounts, events, path.as_posix())


LABEL_MODES: dict[str, str] = {
    "any_event": (
        "Engine parity: any SBA/USAspending/job-posting event occurs within the horizon, "
        "including later lifecycle stages of an instance that already existed."
    ),
    "new_instance": (
        "The first event of a signal instance not seen before as_of occurs within the horizon "
        "(a genuinely new loan/award/posting)."
    ),
}


POPULATIONS: dict[str, str] = {
    "all": "Every sampled account; source coverage alone identifies SBA borrowers.",
    "sba_history": (
        "Only (account, as_of) rows with at least one SBA event already visible at as_of: "
        "does a further/new loan follow for existing borrowers?"
    ),
    "linked": "Only accounts with both SBA and OSHA events somewhere in the corpus.",
}


def population_filter(
    events: Iterable[NormalizedEvent], population: str
) -> Callable[[str, datetime], bool]:
    """Row predicate (account_id, as_of) implementing a POPULATIONS entry."""
    if population not in POPULATIONS:
        raise ValueError(f"unknown population {population!r}; choose from {sorted(POPULATIONS)}")
    if population == "all":
        return lambda account_id, as_of: True
    first_sba: dict[str, datetime] = {}
    types_by_account: dict[str, set[str]] = {}
    for event in events:
        types_by_account.setdefault(event.account_id, set()).add(event.signal_type)
        if event.signal_type == "sba_loan":
            current = first_sba.get(event.account_id)
            if current is None or event.available_at < current:
                first_sba[event.account_id] = event.available_at
    if population == "sba_history":
        return lambda account_id, as_of: account_id in first_sba and first_sba[account_id] <= as_of
    linked = {
        account_id
        for account_id, types in types_by_account.items()
        if {"sba_loan", "osha_injury_summary"} <= types
    }
    return lambda account_id, as_of: account_id in linked


def pristine_labels(
    events: Iterable[NormalizedEvent],
    dates: Sequence[datetime],
    horizon_days: int,
    *,
    label_mode: str = "any_event",
) -> dict[tuple[str, str], int]:
    """The future-event label for every (account, as_of), computed once from raw events."""
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unknown label mode {label_mode!r}; choose from {sorted(LABEL_MODES)}")
    by_account: dict[str, list[datetime]] = {}
    if label_mode == "any_event":
        for event in events:
            if event.signal_type in LABEL_SIGNAL_TYPES:
                by_account.setdefault(event.account_id, []).append(event.occurred_at)
    else:
        first_seen: dict[str, tuple[str, datetime]] = {}
        for event in events:
            if event.signal_type not in LABEL_SIGNAL_TYPES:
                continue
            current = first_seen.get(event.signal_instance_id)
            if current is None or event.occurred_at < current[1]:
                first_seen[event.signal_instance_id] = (event.account_id, event.occurred_at)
        for account_id, occurred in first_seen.values():
            by_account.setdefault(account_id, []).append(occurred)
    labels: dict[tuple[str, str], int] = {}
    for as_of in dates:
        horizon = as_of + timedelta(days=horizon_days)
        for account_id, occurrences in by_account.items():
            labels[(account_id, as_of.isoformat())] = int(
                any(as_of < occurred <= horizon for occurred in occurrences)
            )
    return labels


def run_architecture_ablation(
    root: Path,
    out_dir: Path,
    *,
    corpus_path: Path | None = None,
    max_accounts: int = 10_000,
    arms: Sequence[str] = DEFAULT_ARCHITECTURE_ARMS,
    models: Sequence[str] = DEFAULT_ARCHITECTURE_MODELS,
    seeds: int = 1,
    base_seed: int = 7,
    bootstrap_samples: int = 200,
    precision_k: int = 100,
    profile: BenchmarkProfile = LAPTOP_PROFILE,
    report_suffix: str = "v1",
    label_mode: str = "any_event",
    population: str = "all",
    horizon_days: int | None = None,
    log: Log = None,
) -> dict[str, object]:
    arm_keys = _resolve(arms, ARCHITECTURE_ARMS, "arms")
    model_keys = _resolve(models, MODEL_SPECS, "models")
    corpus = load_manufacturing_corpus(corpus_path or root / "data" / "public.duckdb", max_accounts)
    registry = SignalRegistry.from_directory(root / "signals")
    context = load_decision_context(root / "configs" / "side-manufacturing.yaml")
    horizon = horizon_days or context.goal.horizon_days
    labels = pristine_labels(corpus.events, ARCHITECTURE_DATES, horizon, label_mode=label_mode)
    in_population = population_filter(corpus.events, population)
    loaded_accounts = corpus.account_count
    if population != "all":
        # Accounts that can never enter the population need not be featurized at all; for
        # sba_history this cuts the public corpus from 130k accounts to the ~31k with SBA events.
        eligible = {
            account.account_id
            for account in corpus.accounts
            if any(in_population(account.account_id, date) for date in ARCHITECTURE_DATES)
        }
        corpus = ManufacturingCorpus(
            [account for account in corpus.accounts if account.account_id in eligible],
            [event for event in corpus.events if event.account_id in eligible],
            corpus.source,
        )
    if log:
        log(
            f"[ablate] corpus {loaded_accounts} accounts loaded, {corpus.account_count} in "
            f"population={population}, {corpus.event_count} events, types={corpus.signal_types}, "
            f"labelMode={label_mode}, horizonDays={horizon}, "
            f"positives={sum(labels.values())}/{len(labels)}"
        )
    materializations: dict[str, tuple[list[TrainingRow], SnapshotIndex, int]] = {}

    def materialize(
        arm: ArchitectureArm, seed: int
    ) -> tuple[list[TrainingRow], SnapshotIndex, int]:
        cache_key = arm.variant if arm.events_fn is None else f"{arm.variant}:{seed}"
        if cache_key in materializations:
            return materializations[cache_key]
        started = time.perf_counter()
        arm_context = arm.context_fn(context) if arm.context_fn else context
        arm_registry = arm.registry_fn(registry) if arm.registry_fn else registry
        arm_events = arm.events_fn(corpus.events, seed) if arm.events_fn else corpus.events
        materialization = build_training_materialization(
            accounts=corpus.accounts,
            events=arm_events,
            dates=ARCHITECTURE_DATES,
            context=arm_context,
            registry=arm_registry,
            label_kind="public_future_event",
            include_story_features=arm.story_features,
        )
        # Labels always come from the pristine corpus; overrides are expected only when
        # label_mode differs from the engine's any_event rule.
        mismatches = 0
        rows: list[TrainingRow] = []
        for row in materialization.rows:
            pristine = labels.get((row.account_id, row.as_of.isoformat()), 0)
            mismatches += int(pristine != row.label)
            rows.append(
                row if pristine == row.label else row.model_copy(update={"label": pristine})
            )
        snapshots = {(s.account_id, s.as_of.isoformat()): s for s in materialization.snapshots}
        if log:
            feature_names = {name for row in rows for name in row.features}
            log(
                f"[ablate] materialized variant={cache_key} rows={len(rows)} "
                f"features={len(feature_names)} labelOverrides={mismatches} "
                f"positives={sum(row.label for row in rows)} "
                f"in {time.perf_counter() - started:.1f}s"
            )
        materializations[cache_key] = (rows, snapshots, mismatches)
        return materializations[cache_key]

    cells: list[dict[str, object]] = []
    for arm_key in arm_keys:
        arm = ARCHITECTURE_ARMS[arm_key]
        for seed in _seeds(seeds, base_seed):
            rows, snapshots, mismatches = materialize(arm, seed)
            rows = [row for row in rows if in_population(row.account_id, row.as_of)]
            masked_rows, masked_snapshots, feature_count = apply_feature_mask(
                rows, snapshots, arm.keep
            )
            train, test = _split_point_in_time(masked_rows, seed)
            train = apply_post_split_control(train, arm.post_split_control, seed)
            for model_key in model_keys:
                scorer = MODEL_SPECS[model_key].build(seed, profile)
                cell = evaluate_cell(
                    scorer,
                    train,
                    test,
                    masked_snapshots,
                    precision_k=precision_k,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
                cell.update(
                    {
                        "dataset": "manufacturing-public",
                        "arm": arm_key,
                        "armDescription": arm.description,
                        "modelKey": model_key,
                        "protocol": "point_in_time",
                        "mask": arm_key,
                        "control": arm.post_split_control,
                        "maskedFeatureCount": feature_count,
                        "labelMode": label_mode,
                        "population": population,
                        "labelOverrides": mismatches,
                    }
                )
                cells.append(cell)
                if log:
                    log(_format_cell(cell))
    aggregates = aggregate_cells(cells, ("arm", "modelKey"))
    report = {
        "schemaVersion": 1,
        "kind": "architecture-ablation",
        "dataset": "manufacturing-public",
        "generatedAt": datetime.now(UTC).isoformat(),
        "profile": profile.describe(),
        "hardware": hardware_summary(),
        "corpus": {
            "source": corpus.source,
            "accountsLoaded": loaded_accounts,
            "accounts": corpus.account_count,
            "events": corpus.event_count,
            "signalTypes": corpus.signal_types,
            "dates": [date.isoformat() for date in ARCHITECTURE_DATES],
            "horizonDays": horizon,
            "engineHorizonDays": context.goal.horizon_days,
            "labelMode": label_mode,
            "labelModeDescription": LABEL_MODES[label_mode],
            "population": population,
            "populationDescription": POPULATIONS[population],
            "labelPositives": int(sum(labels.values())),
            "labelRows": len(labels),
        },
        "arms": {key: ARCHITECTURE_ARMS[key].description for key in arm_keys},
        "grid": {
            "models": model_keys,
            "seeds": list(_seeds(seeds, base_seed)),
            "bootstrapSamples": bootstrap_samples,
            "precisionK": precision_k,
        },
        "cells": cells,
        "aggregates": aggregates,
    }
    report_path = out_dir / f"architecture-ablation-{report_suffix}.json"
    _write_report(report_path, report)
    return {"status": "ok", "reportPath": report_path.as_posix(), "cells": len(cells)}
