"""Bayesian-smoothed signal portfolio proposals for the next policy version."""

from __future__ import annotations

import math

from .contracts import PortfolioWeight, SignalPolicy, TrainingRow


def propose_portfolio_weights(
    rows: list[TrainingRow],
    policy: SignalPolicy,
    *,
    prior_strength: float = 25,
    ema_alpha: float = 0.25,
) -> list[PortfolioWeight]:
    if not rows:
        return []
    baseline_rate = sum(row.label for row in rows) / len(rows)
    proposals = []
    for entry in policy.entries:
        feature = f"signal.{entry.signal_type}.active_count"
        exposed = [row for row in rows if row.features.get(feature, 0) > 0]
        positives = sum(row.label for row in exposed)
        exposed_rate = positives / len(exposed) if exposed else baseline_rate
        prior_positive = baseline_rate * prior_strength
        posterior_rate = (positives + prior_positive) / (len(exposed) + prior_strength)
        denominator = max(baseline_rate, 1e-6)
        smoothed_lift = max(posterior_rate / denominator, 0)
        evidence = len(exposed) / (len(exposed) + prior_strength)
        target = 1 / (1 + math.exp(-math.log(max(smoothed_lift, 1e-6))))
        target = 0.5 + (target - 0.5) * evidence * 2
        proposed = (1 - ema_alpha) * entry.initial_weight + ema_alpha * target
        proposed = min(max(proposed, 0.05), 1.0)
        proposals.append(
            PortfolioWeight(
                signal_type=entry.signal_type,
                observations=len(exposed),
                positives=positives,
                baseline_rate=baseline_rate,
                exposed_rate=exposed_rate,
                smoothed_lift=smoothed_lift,
                previous_weight=entry.initial_weight,
                proposed_weight=proposed,
            )
        )
    return proposals
