"""Component scoring utilities for Security-Aware Selective Compression.

The public release operates on aggregate component-level deltas. It does not
read prompts, model generations, adapter weights, or trigger strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for the TH/H/TB/B counterfactual objective.

    Deltas are interpreted as values after removing a component subtracted from
    the reference adapter behavior. Thus positive ``delta_th`` means component
    removal reduced triggered-harmful attack success, and positive ``delta_h``
    means removal improved harmful-prompt refusal. Positive ``delta_tb`` and
    ``delta_b`` are treated as refusal costs on benign prompts.
    """

    th: float = 1.0
    h: float = 0.25
    tb: float = 0.5
    b: float = 0.25


@dataclass(frozen=True)
class ComponentScore:
    """One LoRA component with counterfactual behavioral deltas."""

    component_id: str
    params: float
    delta_th: float
    delta_h: float = 0.0
    delta_tb: float = 0.0
    delta_b: float = 0.0
    layer: str | None = None
    projection: str | None = None

    def score(self, weights: ScoreWeights = ScoreWeights()) -> float:
        return (
            weights.th * self.delta_th
            + weights.h * self.delta_h
            - weights.tb * max(self.delta_tb, 0.0)
            - weights.b * max(self.delta_b, 0.0)
        )


def rank_components(
    components: Iterable[ComponentScore],
    weights: ScoreWeights = ScoreWeights(),
) -> list[tuple[ComponentScore, float]]:
    """Return components sorted by descending SAC score."""

    scored = [(component, component.score(weights)) for component in components]
    return sorted(scored, key=lambda item: (item[1], -item[0].params, item[0].component_id), reverse=True)


def select_budget(
    components: Iterable[ComponentScore],
    budget: float,
    weights: ScoreWeights = ScoreWeights(),
) -> list[tuple[ComponentScore, float]]:
    """Select components until the requested parameter budget is reached.

    ``budget`` may be a fraction in ``(0, 1]`` or an absolute parameter count.
    Fractional budgets are interpreted relative to the total ``params`` mass.
    """

    ranked = rank_components(components, weights)
    total_params = sum(max(component.params, 0.0) for component, _ in ranked)
    if total_params <= 0:
        raise ValueError("component params must sum to a positive value")

    target = budget * total_params if 0 < budget <= 1 else budget
    if target <= 0:
        raise ValueError("budget must be positive")

    selected: list[tuple[ComponentScore, float]] = []
    running = 0.0
    for component, score in ranked:
        if running >= target:
            break
        selected.append((component, score))
        running += max(component.params, 0.0)
    return selected
