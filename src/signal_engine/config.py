"""Load and validate versioned product, ICP, goal, and signal-policy fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .contracts import GoalSpec, IcpSpec, ProductSpec, SignalPolicy


@dataclass(frozen=True)
class DecisionContext:
    product: ProductSpec
    icp: IcpSpec
    goal: GoalSpec
    signal_policy: SignalPolicy


def load_decision_context(path: Path) -> DecisionContext:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DecisionContext(
        product=ProductSpec.model_validate(raw["product"]),
        icp=IcpSpec.model_validate(raw["icp"]),
        goal=GoalSpec.model_validate(raw["goal"]),
        signal_policy=SignalPolicy.model_validate(raw["signal_policy"]),
    )
