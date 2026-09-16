"""Uplift meta-learners and evaluation for randomized-treatment outcome data.

Everything here works on plain matrices so any base learner (LightGBM, logistic,
random forest, XGBoost, CatBoost, TabPFN) can be plugged into S-, T-, X-, DR-, and
R-learners, and so the same Qini/AUUC/uplift@k/IPW metrics can be bootstrapped.
The randomized design is assumed: the propensity is a constant estimated from the
training treatment share, not modeled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import TabPfnConfig


@dataclass(frozen=True)
class BaseLearner:
    """A classifier/regressor pair sharing one configuration."""

    key: str
    classifier: Callable[[int], Any]
    regressor: Callable[[int], Any] | None
    supports_sample_weight: bool = True
    fill_nan: bool = False
    max_training_rows: int | None = None

    def describe(self) -> dict[str, object]:
        return {
            "key": self.key,
            "hasRegressor": self.regressor is not None,
            "supportsSampleWeight": self.supports_sample_weight,
            "maxTrainingRows": self.max_training_rows,
        }


def _lightgbm_pair(seed: int, **overrides: Any) -> tuple[Any, Any]:
    from lightgbm import LGBMClassifier, LGBMRegressor

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
    return LGBMClassifier(**params), LGBMRegressor(**params)


class _TabPfnEstimator:
    """sklearn-like wrapper that applies the TabPFN execution profile and row cap."""

    def __init__(self, kind: str, config: TabPfnConfig, seed: int):
        self.kind = kind
        self.config = config
        self.seed = seed
        self.model: Any = None
        self.rows_used = 0

    def _build(self) -> Any:
        if self.kind == "classifier":
            from tabpfn import TabPFNClassifier

            return TabPFNClassifier(
                random_state=self.seed,
                n_estimators=self.config.n_estimators,
                device=self.config.device,
                ignore_pretraining_limits=self.config.ignore_pretraining_limits,
                show_progress_bar=False,
            )
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            random_state=self.seed,
            n_estimators=self.config.n_estimators,
            device=self.config.device,
            ignore_pretraining_limits=self.config.ignore_pretraining_limits,
            show_progress_bar=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Any:
        if sample_weight is not None:
            raise RuntimeError("TabPFN base learners do not support sample weights")
        cap = self.config.max_training_rows
        if cap is not None and len(y) > cap:
            rng = np.random.default_rng(self.seed)
            keep = np.sort(rng.choice(len(y), cap, replace=False))
            X, y = X[keep], y[keep]
        self.model = self._build()
        self.model.fit(X, y)
        self.rows_used = len(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))


def base_learners(tabpfn: TabPfnConfig) -> dict[str, BaseLearner]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def xgb_classifier(seed: int) -> Any:
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

    def xgb_regressor(seed: int) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
        )

    def cat_classifier(seed: int) -> Any:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6, random_seed=seed, verbose=False
        )

    def cat_regressor(seed: int) -> Any:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6, random_seed=seed, verbose=False
        )

    return {
        "lightgbm": BaseLearner(
            "lightgbm",
            classifier=lambda seed: _lightgbm_pair(seed)[0],
            regressor=lambda seed: _lightgbm_pair(seed)[1],
        ),
        "lightgbm-regularized": BaseLearner(
            "lightgbm-regularized",
            classifier=lambda seed: _lightgbm_pair(
                seed,
                n_estimators=400,
                learning_rate=0.02,
                num_leaves=7,
                min_child_samples=100,
                reg_lambda=5.0,
            )[0],
            regressor=lambda seed: _lightgbm_pair(
                seed,
                n_estimators=400,
                learning_rate=0.02,
                num_leaves=7,
                min_child_samples=100,
                reg_lambda=5.0,
            )[1],
        ),
        "logistic": BaseLearner(
            "logistic",
            classifier=lambda seed: make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)
            ),
            regressor=lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            supports_sample_weight=False,
            fill_nan=True,
        ),
        "random-forest": BaseLearner(
            "random-forest",
            classifier=lambda seed: RandomForestClassifier(
                n_estimators=400, min_samples_leaf=20, n_jobs=-1, random_state=seed
            ),
            regressor=lambda seed: RandomForestRegressor(
                n_estimators=400, min_samples_leaf=20, n_jobs=-1, random_state=seed
            ),
            fill_nan=True,
        ),
        "xgboost": BaseLearner("xgboost", classifier=xgb_classifier, regressor=xgb_regressor),
        "catboost": BaseLearner("catboost", classifier=cat_classifier, regressor=cat_regressor),
        "tabpfn": BaseLearner(
            "tabpfn",
            classifier=lambda seed: _TabPfnEstimator("classifier", tabpfn, seed),
            regressor=lambda seed: _TabPfnEstimator("regressor", tabpfn, seed),
            supports_sample_weight=False,
            max_training_rows=tabpfn.max_training_rows,
        ),
    }


# Meta-learners -------------------------------------------------------------------------


@dataclass(frozen=True)
class UpliftFit:
    scores: np.ndarray
    detail: dict[str, object]


def _proba(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def _prepare(X: np.ndarray, learner: BaseLearner) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0) if learner.fill_nan else X


def _fit_classifier(learner: BaseLearner, seed: int, X: np.ndarray, y: np.ndarray) -> Any:
    if len(np.unique(y)) < 2:
        raise RuntimeError("a base classifier received a single-class training set")
    model = learner.classifier(seed)
    model.fit(X, y)
    return model


def _fit_regressor(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Any:
    if learner.regressor is None:
        raise RuntimeError(f"{learner.key} has no regressor for pseudo-outcome regression")
    model = learner.regressor(seed)
    if sample_weight is not None:
        if not learner.supports_sample_weight:
            raise RuntimeError(f"{learner.key} does not support sample weights (R-learner)")
        model.fit(X, target, sample_weight=sample_weight)
    else:
        model.fit(X, target)
    return model


def _cross_fitted_outcomes(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    folds: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold mu1(x), mu0(x), m(x) on the training set for DR/R pseudo-outcomes."""
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, folds, size=len(y))
    mu1 = np.zeros(len(y))
    mu0 = np.zeros(len(y))
    m = np.zeros(len(y))
    for fold in range(folds):
        fit_mask = fold_of != fold
        apply_mask = ~fit_mask
        treated = fit_mask & (t == 1)
        control = fit_mask & (t == 0)
        mu1[apply_mask] = _proba(
            _fit_classifier(learner, seed + fold, X[treated], y[treated]), X[apply_mask]
        )
        mu0[apply_mask] = _proba(
            _fit_classifier(learner, seed + fold, X[control], y[control]), X[apply_mask]
        )
        m[apply_mask] = _proba(
            _fit_classifier(learner, seed + fold, X[fit_mask], y[fit_mask]), X[apply_mask]
        )
    return mu1, mu0, m


def s_learner(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    propensity: float,
) -> UpliftFit:
    del propensity
    X, X_test = _prepare(X, learner), _prepare(X_test, learner)
    model = _fit_classifier(learner, seed, np.column_stack([X, t]), y)
    ones = np.ones((len(X_test), 1))
    scores = _proba(model, np.column_stack([X_test, ones])) - _proba(
        model, np.column_stack([X_test, ones * 0])
    )
    return UpliftFit(scores, {"learner": "s", "trainRows": len(y)})


def t_learner(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    propensity: float,
) -> UpliftFit:
    del propensity
    X, X_test = _prepare(X, learner), _prepare(X_test, learner)
    treated = _fit_classifier(learner, seed, X[t == 1], y[t == 1])
    control = _fit_classifier(learner, seed, X[t == 0], y[t == 0])
    scores = _proba(treated, X_test) - _proba(control, X_test)
    return UpliftFit(
        scores,
        {"learner": "t", "treatedRows": int((t == 1).sum()), "controlRows": int((t == 0).sum())},
    )


def x_learner(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    propensity: float,
) -> UpliftFit:
    X, X_test = _prepare(X, learner), _prepare(X_test, learner)
    treated = _fit_classifier(learner, seed, X[t == 1], y[t == 1])
    control = _fit_classifier(learner, seed, X[t == 0], y[t == 0])
    imputed_treated = y[t == 1] - _proba(control, X[t == 1])
    imputed_control = _proba(treated, X[t == 0]) - y[t == 0]
    tau_treated = _fit_regressor(learner, seed, X[t == 1], imputed_treated)
    tau_control = _fit_regressor(learner, seed, X[t == 0], imputed_control)
    scores = propensity * np.asarray(tau_control.predict(X_test)) + (1 - propensity) * np.asarray(
        tau_treated.predict(X_test)
    )
    return UpliftFit(scores, {"learner": "x", "propensity": propensity})


def dr_learner(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    propensity: float,
) -> UpliftFit:
    X, X_test = _prepare(X, learner), _prepare(X_test, learner)
    mu1, mu0, _ = _cross_fitted_outcomes(learner, seed, X, t, y)
    mu_t = np.where(t == 1, mu1, mu0)
    weights = (t - propensity) / (propensity * (1 - propensity))
    pseudo = mu1 - mu0 + weights * (y - mu_t)
    model = _fit_regressor(learner, seed, X, pseudo)
    return UpliftFit(
        np.asarray(model.predict(X_test)),
        {"learner": "dr", "crossFitFolds": 2, "propensity": propensity},
    )


def r_learner(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    propensity: float,
) -> UpliftFit:
    X, X_test = _prepare(X, learner), _prepare(X_test, learner)
    _, _, m = _cross_fitted_outcomes(learner, seed, X, t, y)
    residual_t = t - propensity
    target = (y - m) / residual_t
    model = _fit_regressor(learner, seed, X, target, sample_weight=residual_t**2)
    return UpliftFit(
        np.asarray(model.predict(X_test)),
        {"learner": "r", "crossFitFolds": 2, "propensity": propensity},
    )


def response_propensity(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    propensity: float,
) -> UpliftFit:
    """Non-causal baseline: rank by P(outcome) ignoring treatment."""
    del t, propensity
    X, X_test = _prepare(X, learner), _prepare(X_test, learner)
    model = _fit_classifier(learner, seed, X, y)
    return UpliftFit(_proba(model, X_test), {"learner": "response_propensity"})


def random_targeting(
    learner: BaseLearner,
    seed: int,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    propensity: float,
) -> UpliftFit:
    del learner, X, t, y, propensity
    return UpliftFit(np.random.default_rng(seed).random(len(X_test)), {"learner": "random"})


UpliftLearner = Callable[
    [BaseLearner, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], UpliftFit
]
UPLIFT_LEARNERS: Mapping[str, UpliftLearner] = {
    "random": random_targeting,
    "response-propensity": response_propensity,
    "s-learner": s_learner,
    "t-learner": t_learner,
    "x-learner": x_learner,
    "dr-learner": dr_learner,
    "r-learner": r_learner,
}


# Metrics -------------------------------------------------------------------------------


def _cumulative_counts(
    order: np.ndarray, t: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    treated = (t[order] == 1).astype(float)
    control = 1.0 - treated
    n_t = np.cumsum(treated)
    n_c = np.cumsum(control)
    y_t = np.cumsum(y[order] * treated)
    y_c = np.cumsum(y[order] * control)
    return n_t, n_c, y_t, y_c


def qini_curve(scores: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Incremental responders after targeting the top-k, k=1..n (Radcliffe's Qini curve)."""
    order = np.argsort(-scores, kind="stable")
    n_t, n_c, y_t, y_c = _cumulative_counts(order, t, y)
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    return np.asarray(y_t - y_c * ratio)


def uplift_curve(scores: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(treated rate - control rate) x k for the top-k, k=1..n (the sklift uplift curve)."""
    order = np.argsort(-scores, kind="stable")
    n_t, n_c, y_t, y_c = _cumulative_counts(order, t, y)
    rate_t = np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t > 0)
    rate_c = np.divide(y_c, n_c, out=np.zeros_like(y_c), where=n_c > 0)
    return np.asarray((rate_t - rate_c) * np.arange(1, len(order) + 1))


def _area(curve: np.ndarray) -> float:
    return float(np.trapezoid(curve, dx=1.0))


def _normalized_area(
    curve_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    scores: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """(unscaled area above random per row, sklift-style area normalized by the perfect model)."""
    curve = curve_fn(scores, t, y)
    n = len(curve)
    random_line = np.linspace(curve[-1] / n, curve[-1], n)
    perfect_scores = np.where(t == 1, y, 1 - y).astype(float)
    perfect = curve_fn(perfect_scores, t, y)
    model_area = _area(curve) - _area(random_line)
    perfect_area = _area(perfect) - _area(random_line)
    normalized = model_area / perfect_area if perfect_area > 0 else 0.0
    return model_area / n, float(normalized)


def uplift_at_fraction(
    scores: np.ndarray, t: np.ndarray, y: np.ndarray, fraction: float
) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    top = order[: max(1, int(len(order) * fraction))]
    treated = top[t[top] == 1]
    control = top[t[top] == 0]
    rate_t = float(y[treated].mean()) if len(treated) else 0.0
    rate_c = float(y[control].mean()) if len(control) else 0.0
    return {
        "treatedRate": rate_t,
        "controlRate": rate_c,
        "uplift": rate_t - rate_c,
        "treated": len(treated),
        "control": len(control),
    }


def ipw_policy_value(policy: np.ndarray, t: np.ndarray, y: np.ndarray, propensity: float) -> float:
    """Inverse-propensity estimate of the outcome rate if `policy` decided who is treated."""
    treated_match = policy & (t == 1)
    control_match = (~policy) & (t == 0)
    value = np.where(
        treated_match, y / propensity, np.where(control_match, y / (1 - propensity), 0.0)
    )
    return float(value.mean())


def uplift_metrics(
    scores: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    propensity: float,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> dict[str, object]:
    qini_unscaled, qini_normalized = _normalized_area(qini_curve, scores, t, y)
    auuc_unscaled, auuc_normalized = _normalized_area(uplift_curve, scores, t, y)
    ate = float(y[t == 1].mean() - y[t == 0].mean())
    top20 = np.zeros(len(scores), dtype=bool)
    top20[np.argsort(-scores, kind="stable")[: max(1, int(len(scores) * 0.2))]] = True
    positive = scores > 0
    return {
        "testRows": len(scores),
        "testAte": ate,
        "qiniCoefficient": qini_unscaled,
        "qiniNormalized": qini_normalized,
        "auuc": auuc_unscaled,
        "auucNormalized": auuc_normalized,
        "upliftAt": {f"{int(f * 100)}": uplift_at_fraction(scores, t, y, f) for f in fractions},
        "policyPositiveShare": float(positive.mean()),
        "ipwPolicyPositive": ipw_policy_value(positive, t, y, propensity),
        "ipwPolicyTop20": ipw_policy_value(top20, t, y, propensity),
        "ipwContactAll": ipw_policy_value(np.ones(len(scores), dtype=bool), t, y, propensity),
        "ipwContactNone": ipw_policy_value(np.zeros(len(scores), dtype=bool), t, y, propensity),
    }


def bootstrap_uplift(
    scores: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    propensity: float,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Percentile 95% intervals over test-row resamples for the headline uplift metrics."""
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"qiniNormalized": [], "upliftAt20": [], "ipwPolicyTop20": []}
    n = len(scores)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        ts, ys, ss = t[idx], y[idx], scores[idx]
        if ys[ts == 1].sum() == 0 or (ts == 0).sum() == 0 or (ts == 1).sum() == 0:
            continue
        _, qini_normalized = _normalized_area(qini_curve, ss, ts, ys)
        draws["qiniNormalized"].append(qini_normalized)
        draws["upliftAt20"].append(uplift_at_fraction(ss, ts, ys, 0.2)["uplift"])
        top = np.zeros(n, dtype=bool)
        top[np.argsort(-ss, kind="stable")[: max(1, int(n * 0.2))]] = True
        draws["ipwPolicyTop20"].append(ipw_policy_value(top, ts, ys, propensity))
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in draws.items()
        if values
    }
