"""Swappable heuristic, LightGBM, and TabPFN scorers plus honest benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib import metadata
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    roc_auc_score,
)

from .contracts import AccountSnapshot, BenchmarkResult, ModelScore, TrainingRow


def _matrix(
    feature_rows: list[dict[str, float]],
    columns: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    resolved = columns or tuple(sorted({name for features in feature_rows for name in features}))
    return (
        np.asarray(
            [[features.get(column, np.nan) for column in resolved] for features in feature_rows],
            dtype=np.float64,
        ),
        resolved,
    )


class Scorer(ABC):
    model_id: str

    @abstractmethod
    def fit(self, rows: list[TrainingRow]) -> None:
        raise NotImplementedError

    @abstractmethod
    def score(self, snapshots: list[AccountSnapshot]) -> list[ModelScore]:
        raise NotImplementedError

    def runtime_info(self) -> dict[str, object]:
        """Execution evidence (device, rows actually used, retries) for benchmark reports."""
        return {}


class HeuristicScorer(Scorer):
    model_id = "heuristic-v1"

    def fit(self, rows: list[TrainingRow]) -> None:
        del rows

    def score(self, snapshots: list[AccountSnapshot]) -> list[ModelScore]:
        results = []
        for snapshot in snapshots:
            weighted = snapshot.features.get("signals.weighted_sum", 0.0)
            sequence = sum(
                value
                for name, value in snapshot.features.items()
                if name.startswith("sequence.") and name.endswith(".weighted")
            )
            fit = snapshot.features.get("fit.score", 0.0)
            corroboration = snapshot.features.get("signals.corroboration", 0.0)
            raw = (
                0.35 * fit
                + 0.35 * (1 - np.exp(-weighted))
                + 0.2 * (1 - np.exp(-sequence))
                + 0.1 * corroboration
            )
            factors = sorted(
                (
                    ("fit.score", 0.35 * fit),
                    ("signals.weighted_sum", 0.35 * (1 - np.exp(-weighted))),
                    ("sequences", 0.2 * (1 - np.exp(-sequence))),
                    ("signals.corroboration", 0.1 * corroboration),
                ),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            results.append(
                ModelScore(
                    model_id=self.model_id,
                    account_id=snapshot.account_id,
                    score=float(np.clip(raw, 0, 1)),
                    top_factors=tuple((name, float(value)) for name, value in factors),
                    calibrated=False,
                )
            )
        return results


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(clipped / (1 - clipped)))


CalibrationMode = Literal["time_series_cv", "holdout", "none"]


class LightGbmScorer(Scorer):
    """Balanced LightGBM with Platt calibration that can never invert the ranking.

    ``time_series_cv`` (default) fits expanding-window models, calibrates on their
    out-of-fold predictions for later slices, then refits on every row. ``holdout`` is
    the previous behaviour (train on the oldest 80%, calibrate on the newest 20%), kept
    for comparison: on drifting data it trained on stale rows and, on UCI Bank Marketing,
    the calibrator learned a negative slope and flipped an inverted model into a
    plausible-looking score. Any calibrator with a non-positive slope is now rejected.
    """

    model_id = "lightgbm-v1"

    def __init__(
        self,
        random_seed: int = 7,
        *,
        calibration: CalibrationMode = "time_series_cv",
        calibration_fraction: float = 0.2,
        model_id: str | None = None,
    ):
        if not 0 <= calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in [0, 1)")
        if model_id:
            self.model_id = model_id
        self.calibration = calibration
        self.calibration_fraction = calibration_fraction
        self.calibration_note: str | None = None
        self.columns: tuple[str, ...] = ()
        self.calibrator: LogisticRegression | None = None
        self.training_rows_used = 0
        self.model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_seed,
            class_weight="balanced",
            verbosity=-1,
        )

    def fit(self, rows: list[TrainingRow]) -> None:
        if len({row.label for row in rows}) < 2:
            raise ValueError("LightGBM requires both positive and negative labels")
        matrix, self.columns = _matrix([row.features for row in rows])
        labels = np.asarray([row.label for row in rows])
        order = np.argsort(np.asarray([row.as_of.timestamp() for row in rows], dtype=np.float64))
        self.calibrator = None
        self.calibration_note = None
        oof_logits: np.ndarray | None = None
        oof_labels: np.ndarray | None = None
        if self.calibration == "holdout" and self.calibration_fraction > 0:
            boundary = max(1, min(len(rows) - 1, int(len(rows) * (1 - self.calibration_fraction))))
            train_indices, held_out = order[:boundary], order[boundary:]
            self.model.fit(matrix[train_indices], labels[train_indices])
            self.training_rows_used = len(train_indices)
            oof_logits = _logits(np.asarray(self.model.predict_proba(matrix[held_out]))[:, 1])
            oof_labels = labels[held_out]
        else:
            if self.calibration == "time_series_cv":
                oof_logits, oof_labels = self._time_series_out_of_fold(matrix, labels, order)
            self.model.fit(matrix, labels)
            self.training_rows_used = len(rows)
        if self.calibration == "none":
            return
        if oof_logits is None or oof_labels is None or len(set(oof_labels.tolist())) < 2:
            self.calibration_note = "no out-of-fold rows with both classes; raw probabilities"
            return
        calibrator = LogisticRegression(random_state=self.model.random_state)
        calibrator.fit(oof_logits.reshape(-1, 1), oof_labels)
        slope = float(calibrator.coef_[0][0])
        if slope <= 0:
            # Later rows disagree with the fitted ranking. Inverting the scores would hide that
            # drift behind a plausible number, so keep the raw ranking and record the rejection.
            self.calibration_note = f"calibrator rejected: non-positive Platt slope {slope:.3f}"
            return
        self.calibrator = calibrator

    def _time_series_out_of_fold(
        self, matrix: np.ndarray, labels: np.ndarray, order: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expanding-window predictions on later slices; no row is scored by a model that saw it."""
        from sklearn.base import clone

        cuts = [int(len(order) * fraction) for fraction in (0.4, 0.6, 0.8, 1.0)]
        logits: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for start, stop in pairwise(cuts):
            train_indices, fold_indices = order[:start], order[start:stop]
            if len(fold_indices) == 0 or len(set(labels[train_indices].tolist())) < 2:
                continue
            fold_model = clone(self.model)
            fold_model.fit(matrix[train_indices], labels[train_indices])
            logits.append(_logits(np.asarray(fold_model.predict_proba(matrix[fold_indices]))[:, 1]))
            targets.append(labels[fold_indices])
        if not logits:
            return np.empty(0), np.empty(0, dtype=int)
        return np.concatenate(logits), np.concatenate(targets)

    def score(self, snapshots: list[AccountSnapshot]) -> list[ModelScore]:
        if not self.columns:
            raise RuntimeError("LightGBM scorer is not fitted")
        matrix, _ = _matrix([snapshot.features for snapshot in snapshots], self.columns)
        raw_probabilities = np.asarray(self.model.predict_proba(matrix))[:, 1]
        if self.calibrator is not None:
            logits = np.log(
                np.clip(raw_probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - raw_probabilities, 1e-6, 1)
            )
            probabilities = np.asarray(self.calibrator.predict_proba(logits.reshape(-1, 1)))[:, 1]
        else:
            probabilities = raw_probabilities
        contributions = np.asarray(self.model.booster_.predict(matrix, pred_contrib=True))
        results = []
        for snapshot, probability, row_contributions in zip(
            snapshots, probabilities, contributions, strict=True
        ):
            top_indices = np.argsort(np.abs(row_contributions[:-1]))[::-1][:8]
            factors = tuple(
                (self.columns[index], float(row_contributions[index]))
                for index in top_indices
                if abs(row_contributions[index]) > 1e-9
            )
            results.append(
                ModelScore(
                    model_id=self.model_id,
                    account_id=snapshot.account_id,
                    score=float(probability),
                    top_factors=factors,
                    calibrated=self.calibrator is not None,
                )
            )
        return results

    def runtime_info(self) -> dict[str, object]:
        info: dict[str, object] = {
            "calibration": self.calibration,
            "calibrationFraction": self.calibration_fraction,
            "calibrationNote": self.calibration_note,
            "trainingRowsUsed": self.training_rows_used,
            "calibrated": self.calibrator is not None,
        }
        if self.calibrator is not None:
            # A negative slope means the held-out slice disagreed with the fitted ranking;
            # the calibrator then inverts scores, which is a drift warning, not a fix.
            info["plattCoefficient"] = float(self.calibrator.coef_[0][0])
        return info

    def save_artifact(self, directory: Path) -> dict[str, object]:
        if not self.columns:
            raise RuntimeError("cannot save an unfitted LightGBM scorer")
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "lightgbm-model.txt"
        metadata_path = directory / "lightgbm-metadata.json"
        self.model.booster_.save_model(str(model_path))
        metadata: dict[str, object] = {
            "schemaVersion": 1,
            "modelId": self.model_id,
            "featureColumns": list(self.columns),
            "calibrated": self.calibrator is not None,
        }
        if self.calibrator is not None:
            metadata["calibration"] = {
                "kind": "platt",
                "coefficient": self.calibrator.coef_.tolist(),
                "intercept": self.calibrator.intercept_.tolist(),
            }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "modelPath": model_path.as_posix(),
            "metadataPath": metadata_path.as_posix(),
        }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def hardware_summary() -> dict[str, object]:
    """Describe the machine a benchmark ran on; torch is only imported when installed."""
    summary: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpuCount": os.cpu_count(),
        "packages": {
            name: _package_version(name)
            for name in ("lightgbm", "tabpfn", "torch", "numpy", "scikit-learn")
        },
        "cuda": {"available": False},
    }
    try:
        import torch
    except ImportError:
        return summary
    cuda: dict[str, object] = {"available": bool(torch.cuda.is_available())}
    if cuda["available"]:
        properties = torch.cuda.get_device_properties(0)
        cuda.update(
            {
                "deviceName": torch.cuda.get_device_name(0),
                "deviceCount": torch.cuda.device_count(),
                "totalMemoryGib": round(properties.total_memory / 1024**3, 1),
                "torchCudaVersion": torch.version.cuda,
            }
        )
    summary["cuda"] = cuda
    return summary


@dataclass(frozen=True)
class TabPfnConfig:
    """Execution settings for TabPFN; the defaults are the laptop-speed CPU configuration."""

    max_training_rows: int | None = 600
    n_estimators: int = 2
    device: str = "auto"
    ignore_pretraining_limits: bool = False
    oom_floor_rows: int = 4_000

    def describe(self) -> dict[str, object]:
        return dict(asdict(self))


LAPTOP_TABPFN_CONFIG = TabPfnConfig()
GPU_TABPFN_CONFIG = TabPfnConfig(
    max_training_rows=50_000,
    n_estimators=8,
    device="cuda",
    ignore_pretraining_limits=True,
)


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    text = str(error).lower()
    return type(error).__name__ == "OutOfMemoryError" or (
        "out of memory" in text and "cuda" in text
    )


class TabPfnScorer(Scorer):
    model_id = "tabpfn-local-v2"

    def __init__(
        self,
        random_seed: int = 7,
        max_training_rows: int = 600,
        *,
        config: TabPfnConfig | None = None,
    ):
        self.random_seed = random_seed
        self.config = config or TabPfnConfig(max_training_rows=max_training_rows)
        self.max_training_rows = self.config.max_training_rows
        self.columns: tuple[str, ...] = ()
        self.model: object | None = None
        self._rows: list[TrainingRow] = []
        self._rows_used = 0
        self._attempts: list[dict[str, object]] = []

    def _subsample(self, rows: list[TrainingRow], cap: int | None) -> list[TrainingRow]:
        if cap is None or len(rows) <= cap:
            return rows
        rng = np.random.default_rng(self.random_seed)
        indices = sorted(rng.choice(len(rows), cap, replace=False).tolist())
        return [rows[index] for index in indices]

    def _fit_rows(self, rows: list[TrainingRow]) -> None:
        from tabpfn import TabPFNClassifier

        matrix, self.columns = _matrix([row.features for row in rows])
        model = TabPFNClassifier(
            random_state=self.random_seed,
            n_estimators=self.config.n_estimators,
            device=self.config.device,
            ignore_pretraining_limits=self.config.ignore_pretraining_limits,
            show_progress_bar=False,
        )
        model.fit(matrix, np.asarray([row.label for row in rows]))
        self.model = model
        self._rows_used = len(rows)

    def fit(self, rows: list[TrainingRow]) -> None:
        if os.getenv("TABPFN_ENABLE") != "1":
            raise RuntimeError(
                "TabPFN execution is disabled until its model license is accepted; "
                "set TABPFN_ENABLE=1 after authenticating with PriorLabs"
            )
        try:
            import tabpfn  # noqa: F401
        except ImportError as error:
            raise RuntimeError("TabPFN is optional; install the `tabpfn` project extra") from error
        self._rows = rows
        self._attempts = []
        started = time.perf_counter()
        self._fit_rows(self._subsample(rows, self.config.max_training_rows))
        self._attempts.append(
            {
                "phase": "fit",
                "trainingRows": self._rows_used,
                "seconds": time.perf_counter() - started,
                "status": "ok",
            }
        )

    def _predict(self, matrix: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(
            self.model.predict_proba(matrix)  # type: ignore[attr-defined]
        )[:, 1]

    def score(self, snapshots: list[AccountSnapshot]) -> list[ModelScore]:
        if self.model is None or not self.columns:
            raise RuntimeError("TabPFN scorer is not fitted")
        matrix, _ = _matrix([snapshot.features for snapshot in snapshots], self.columns)
        while True:
            started = time.perf_counter()
            try:
                probabilities = self._predict(matrix)
                break
            except Exception as error:
                if not _is_cuda_out_of_memory(error):
                    raise
                # GPU memory is the binding constraint for in-context learning; halve the
                # context rather than silently failing, and record every attempt.
                reduced = self._rows_used // 2
                retry = reduced >= self.config.oom_floor_rows
                self._attempts.append(
                    {
                        "phase": "score",
                        "trainingRows": self._rows_used,
                        "seconds": time.perf_counter() - started,
                        "status": "cuda_out_of_memory",
                        "retryWithRows": reduced if retry else None,
                    }
                )
                if not retry:
                    raise RuntimeError(
                        "TabPFN ran out of GPU memory even at "
                        f"{self._rows_used} training rows (floor {self.config.oom_floor_rows})"
                    ) from error
                self._release_cuda_memory()
                self._fit_rows(self._subsample(self._rows, reduced))
        return [
            ModelScore(
                model_id=self.model_id,
                account_id=snapshot.account_id,
                score=float(probability),
                uncertainty=float(probability * (1 - probability)),
                calibrated=False,
            )
            for snapshot, probability in zip(snapshots, probabilities, strict=True)
        ]

    def _release_cuda_memory(self) -> None:
        self.model = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def runtime_info(self) -> dict[str, object]:
        info: dict[str, object] = {
            **self.config.describe(),
            "trainingRowsAvailable": len(self._rows),
            "trainingRowsUsed": self._rows_used,
            "attempts": list(self._attempts),
            "tabpfnVersion": _package_version("tabpfn"),
        }
        try:
            import torch
        except ImportError:
            info["cudaAvailable"] = False
            return info
        info["torchVersion"] = torch.__version__
        info["cudaAvailable"] = bool(torch.cuda.is_available())
        # tabpfn>=8 stores the resolved device list as ``devices_``; older builds used ``device_``.
        resolved = getattr(self.model, "devices_", None) or getattr(self.model, "device_", None)
        if resolved is not None:
            info["resolvedDevice"] = str(resolved)
        return info


@dataclass(frozen=True)
class BenchmarkSplit:
    train: list[TrainingRow]
    test: list[TrainingRow]


def point_in_time_split(rows: list[TrainingRow], fraction: float = 0.67) -> BenchmarkSplit:
    if not rows:
        return BenchmarkSplit([], [])
    ordered = sorted(rows, key=lambda row: (row.as_of, row.account_id))
    unique_dates = sorted({row.as_of for row in ordered})
    date_boundary = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * fraction)))
    cutoff = unique_dates[date_boundary]
    test_accounts = {
        row.account_id
        for row in ordered
        if int(hashlib.sha256(row.account_id.encode()).hexdigest()[:8], 16) % 5 == 0
    }
    train = [row for row in ordered if row.as_of < cutoff and row.account_id not in test_accounts]
    test = [row for row in ordered if row.as_of >= cutoff and row.account_id in test_accounts]
    return BenchmarkSplit(train, test)


@dataclass(frozen=True)
class SplitScores:
    """Raw outputs of one fit/score pass so callers can compute any metric or bootstrap."""

    labels: np.ndarray
    probabilities: np.ndarray
    fit_seconds: float
    score_seconds: float


def score_split(
    scorer: Scorer,
    train: list[TrainingRow],
    test: list[TrainingRow],
    snapshots_by_key: dict[tuple[str, str], AccountSnapshot],
) -> SplitScores:
    fit_started = time.perf_counter()
    scorer.fit(train)
    fit_seconds = time.perf_counter() - fit_started
    test_snapshots = [snapshots_by_key[(row.account_id, row.as_of.isoformat())] for row in test]
    score_started = time.perf_counter()
    scores = scorer.score(test_snapshots)
    score_seconds = time.perf_counter() - score_started
    return SplitScores(
        labels=np.asarray([row.label for row in test]),
        probabilities=np.asarray([score.score for score in scores]),
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    precision_k: int,
) -> dict[str, float | None]:
    has_positive = bool(labels.sum() > 0)
    has_both_classes = len(set(labels.tolist())) > 1
    order = np.argsort(probabilities)[::-1][: min(precision_k, len(probabilities))]
    predicted_top = np.zeros_like(labels)
    predicted_top[order] = 1
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)) if has_positive else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "precision_at_k": float(precision_score(labels, predicted_top, zero_division=0)),
    }


def benchmark(
    scorer: Scorer,
    rows: list[TrainingRow],
    snapshots_by_key: dict[tuple[str, str], AccountSnapshot],
    precision_k: int = 50,
) -> BenchmarkResult:
    split = point_in_time_split(rows)
    return evaluate_split(scorer, split.train, split.test, snapshots_by_key, precision_k)


def evaluate_split(
    scorer: Scorer,
    train: list[TrainingRow],
    test: list[TrainingRow],
    snapshots_by_key: dict[tuple[str, str], AccountSnapshot],
    precision_k: int = 50,
) -> BenchmarkResult:
    label_kind = (train or test)[0].label_kind if (train or test) else "unknown"
    if not train or not test:
        return BenchmarkResult(
            model_id=scorer.model_id,
            label_kind=label_kind,
            train_rows=len(train),
            test_rows=len(test),
            pr_auc=None,
            roc_auc=None,
            brier_score=None,
            precision_at_k=None,
            fit_seconds=0,
            score_seconds=0,
            status="failed",
            detail="split produced an empty partition",
        )
    try:
        scored = score_split(scorer, train, test, snapshots_by_key)
        metrics = classification_metrics(scored.labels, scored.probabilities, precision_k)
        has_positive = bool(scored.labels.sum() > 0)
        return BenchmarkResult(
            model_id=scorer.model_id,
            label_kind=label_kind,
            train_rows=len(train),
            test_rows=len(test),
            pr_auc=metrics["pr_auc"],
            roc_auc=metrics["roc_auc"],
            brier_score=metrics["brier_score"],
            precision_at_k=metrics["precision_at_k"],
            fit_seconds=scored.fit_seconds,
            score_seconds=scored.score_seconds,
            status="ok",
            detail=(None if has_positive else "test partition contains no positive proxy outcomes"),
        )
    except RuntimeError as error:
        return BenchmarkResult(
            model_id=scorer.model_id,
            label_kind=label_kind,
            train_rows=len(train),
            test_rows=len(test),
            pr_auc=None,
            roc_auc=None,
            brier_score=None,
            precision_at_k=None,
            fit_seconds=0,
            score_seconds=0,
            status="unavailable",
            detail=str(error),
        )
    except Exception as error:
        return BenchmarkResult(
            model_id=scorer.model_id,
            label_kind=label_kind,
            train_rows=len(train),
            test_rows=len(test),
            pr_auc=None,
            roc_auc=None,
            brier_score=None,
            precision_at_k=None,
            fit_seconds=0,
            score_seconds=0,
            status="failed",
            detail=f"{type(error).__name__}: {error}",
        )
