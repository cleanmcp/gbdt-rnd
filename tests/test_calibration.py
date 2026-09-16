from datetime import UTC, datetime, timedelta

import numpy as np
from sklearn.metrics import roc_auc_score

from signal_engine.contracts import AccountSnapshot, TrainingRow
from signal_engine.hashing import sha256_json
from signal_engine.models import LightGbmScorer


def _rows(n: int, flip_after: float | None, seed: int = 1) -> list[TrainingRow]:
    """Synthetic drift: after `flip_after` of the timeline the feature/label sign inverts."""
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(n):
        x = float(rng.normal())
        z = float(rng.normal())
        late = flip_after is not None and index >= int(n * flip_after)
        signal = -x if late else x
        rows.append(
            TrainingRow(
                account_id=f"acct-{index:05d}",
                as_of=start + timedelta(days=index),
                features={"x": x, "z": z},
                label=int(signal + 0.5 * z > 0.3),
                label_kind="real_outcome",
            )
        )
    return rows


def _snapshots(rows: list[TrainingRow]) -> list[AccountSnapshot]:
    return [
        AccountSnapshot(
            account_id=row.account_id,
            as_of=row.as_of,
            icp_version="t",
            goal_version="t",
            features=row.features,
            coverage={},
            evidence_fact_ids=(),
            snapshot_hash=sha256_json({"a": row.account_id}),
        )
        for row in rows
    ]


def test_holdout_calibrator_is_rejected_when_the_newest_slice_inverts() -> None:
    rows = _rows(1500, flip_after=0.8)
    scorer = LightGbmScorer(7, calibration="holdout")
    scorer.fit(rows)
    assert scorer.calibrator is None
    assert scorer.calibration_note is not None and "rejected" in scorer.calibration_note
    assert scorer.training_rows_used == 1200
    # The raw ranking is preserved on the early regime rather than flipped by the calibrator.
    early = rows[:1000]
    scores = np.asarray([s.score for s in scorer.score(_snapshots(early))])
    assert roc_auc_score([r.label for r in early], scores) > 0.8
    assert scorer.runtime_info()["calibrated"] is False


def test_time_series_cv_calibration_trains_on_all_rows_with_positive_slope() -> None:
    rows = _rows(1500, flip_after=None)
    scorer = LightGbmScorer(7)
    scorer.fit(rows)
    assert scorer.training_rows_used == 1500
    assert scorer.calibrator is not None
    info = scorer.runtime_info()
    assert info["calibration"] == "time_series_cv"
    assert float(info["plattCoefficient"]) > 0  # type: ignore[arg-type]
    scores = scorer.score(_snapshots(rows[-300:]))
    assert all(0.0 <= s.score <= 1.0 and s.calibrated for s in scores)
    assert roc_auc_score([r.label for r in rows[-300:]], [s.score for s in scores]) > 0.8


def test_no_calibration_emits_raw_probabilities() -> None:
    rows = _rows(600, flip_after=None)
    scorer = LightGbmScorer(7, calibration="none", model_id="lightgbm-raw-test")
    scorer.fit(rows)
    assert scorer.calibrator is None and scorer.calibration_note is None
    assert scorer.model_id == "lightgbm-raw-test"
    assert all(not s.calibrated for s in scorer.score(_snapshots(rows[:50])))
