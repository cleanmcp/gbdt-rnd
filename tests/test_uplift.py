import numpy as np
import pytest

from signal_engine.models import TabPfnConfig
from signal_engine.uplift import (
    UPLIFT_LEARNERS,
    base_learners,
    bootstrap_uplift,
    ipw_policy_value,
    qini_curve,
    uplift_at_fraction,
    uplift_metrics,
)


def _rct(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = (rng.random(n) < 0.5).astype(int)
    tau = 0.12 * (X[:, 0] > 0)  # only half the population responds to treatment
    p = 0.05 + tau * t
    y = (rng.random(n) < p).astype(int)
    return X, t, y


def test_perfect_random_and_reference_policies() -> None:
    _, t, y = _rct()
    propensity = float(t.mean())
    perfect = np.where(t == 1, y, 1 - y).astype(float)
    assert uplift_metrics(perfect, t, y, propensity)["qiniNormalized"] == pytest.approx(1.0)
    random_scores = np.random.default_rng(1).random(len(y))
    random_metrics = uplift_metrics(random_scores, t, y, propensity)
    assert abs(float(random_metrics["qiniNormalized"])) < 0.15
    contact_all = ipw_policy_value(np.ones(len(y), dtype=bool), t, y, propensity)
    assert contact_all == pytest.approx(float(y[t == 1].mean()))
    contact_none = ipw_policy_value(np.zeros(len(y), dtype=bool), t, y, propensity)
    assert contact_none == pytest.approx(float(y[t == 0].mean()))
    curve = qini_curve(perfect, t, y)
    assert curve[-1] == pytest.approx(
        y[t == 1].sum() - y[t == 0].sum() * (t == 1).sum() / (t == 0).sum()
    )


def test_meta_learners_recover_heterogeneous_effect_with_lightgbm() -> None:
    X, t, y = _rct()
    test = np.arange(len(y)) % 4 == 0
    train = ~test
    propensity = float(t[train].mean())
    learner = base_learners(TabPfnConfig())["lightgbm"]
    oracle_top = uplift_at_fraction((X[test, 0] > 0).astype(float), t[test], y[test], 0.2)["uplift"]
    for key in ("s-learner", "t-learner", "x-learner", "dr-learner", "r-learner"):
        fitted = UPLIFT_LEARNERS[key](learner, 7, X[train], t[train], y[train], X[test], propensity)
        metrics = uplift_metrics(fitted.scores, t[test], y[test], propensity)
        top20 = metrics["upliftAt"]["20"]["uplift"]  # type: ignore[index]
        assert top20 > float(metrics["testAte"]), key  # type: ignore[arg-type]
        assert top20 > 0.5 * oracle_top, key
        # Normalization is against an oracle that knows individual outcomes, so even a good
        # ranking on a 5% base rate with a 12pp effect scores low in absolute terms.
        assert float(metrics["qiniNormalized"]) > 0.02, key  # type: ignore[arg-type]
    response = UPLIFT_LEARNERS["response-propensity"](
        learner, 7, X[train], t[train], y[train], X[test], propensity
    )
    assert response.scores.min() >= 0 and response.scores.max() <= 1


def test_bootstrap_intervals_bracket_point_estimates() -> None:
    X, t, y = _rct(3000)
    propensity = float(t.mean())
    scores = X[:, 0] + np.random.default_rng(2).normal(scale=0.5, size=len(y))
    metrics = uplift_metrics(scores, t, y, propensity)
    intervals = bootstrap_uplift(scores, t, y, propensity, samples=60, seed=1)
    low, high = intervals["qiniNormalized"]
    assert low <= float(metrics["qiniNormalized"]) <= high  # type: ignore[arg-type]
    low, high = intervals["upliftAt20"]
    assert low <= metrics["upliftAt"]["20"]["uplift"] <= high  # type: ignore[index]


def test_logistic_base_refuses_sample_weights_for_r_learner() -> None:
    X, t, y = _rct(1500)
    learner = base_learners(TabPfnConfig())["logistic"]
    with pytest.raises(RuntimeError):
        UPLIFT_LEARNERS["r-learner"](learner, 7, X, t, y, X[:10], float(t.mean()))
