"""Tiny, dependency-free learned delegation model for the Fast Brain.

The model is intentionally small: a calibrated logistic classifier decides whether
an observation-memory state should be delegated to the Slow Brain. Training data
is synthetic teacher data for this MVP; a research version would replace it with
labeled simulator trajectories or a stronger teacher agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable, Tuple

from .models import BeliefRecord, Freshness


@dataclass(frozen=True)
class DelegationExample:
    features: Tuple[float, float, float, float]
    label: float


class TinyDelegationModel:
    """A four-feature logistic model trained with deterministic SGD."""

    feature_names = (
        "memory_uncertainty",
        "normalized_memory_age",
        "memory_contradiction",
        "unverified_slow_hint",
    )

    def __init__(self, *, threshold: float = 0.65) -> None:
        self.threshold = threshold
        self.weights = [0.0] * len(self.feature_names)
        self.bias = 0.0
        self.is_fitted = False

    @classmethod
    def default(cls) -> "TinyDelegationModel":
        model = cls()
        model.fit(_synthetic_teacher_examples())
        return model

    def fit(
        self,
        examples: Iterable[DelegationExample],
        *,
        epochs: int = 500,
        learning_rate: float = 0.18,
    ) -> None:
        rows = list(examples)
        if not rows:
            raise ValueError("At least one delegation example is required.")
        for _ in range(epochs):
            for example in rows:
                probability = self._sigmoid(self.bias + self._dot(example.features))
                error = probability - example.label
                for index, value in enumerate(example.features):
                    self.weights[index] -= learning_rate * error * value
                self.bias -= learning_rate * error
        self.is_fitted = True

    def predict_probability(self, record: BeliefRecord, current_step: int) -> float:
        if not self.is_fitted:
            raise RuntimeError("TinyDelegationModel must be fitted before prediction.")
        features = self.features_for(record, current_step)
        return self._sigmoid(self.bias + self._dot(features))

    def should_delegate(self, record: BeliefRecord, current_step: int) -> bool:
        return self.predict_probability(record, current_step) >= self.threshold

    @staticmethod
    def features_for(record: BeliefRecord, current_step: int) -> Tuple[float, float, float, float]:
        age = max(0, current_step - record.observed_step)
        return (
            1.0 - record.confidence,
            min(age / 10.0, 1.0),
            float(record.freshness is Freshness.CONTRADICTED),
            float(record.freshness is Freshness.UNVERIFIED),
        )

    def describe(self, record: BeliefRecord, current_step: int) -> str:
        probability = self.predict_probability(record, current_step)
        features = self.features_for(record, current_step)
        return (
            f"delegate_probability={probability:.2f} (threshold={self.threshold:.2f}; "
            f"uncertainty={features[0]:.2f}, age={features[1]:.2f}, "
            f"contradiction={features[2]:.0f}, unverified={features[3]:.0f})"
        )

    def _dot(self, features: Tuple[float, float, float, float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + exp(-value))
        exp_value = exp(value)
        return exp_value / (1.0 + exp_value)


def _synthetic_teacher_examples() -> list[DelegationExample]:
    """Create transparent synthetic labels for a mechanism-validation MVP.

    Teacher rule: contradicting a directly observed memory should trigger Slow
    Brain delegation. Fresh, high-confidence memories should remain on the fast
    path. Old, unverified suggestions become delegation candidates only when
    both age and uncertainty are high.
    """
    examples: list[DelegationExample] = []
    confidences = (0.15, 0.35, 0.55, 0.75, 0.92)
    for freshness in Freshness:
        for confidence in confidences:
            for age in range(10):
                features = (
                    1.0 - confidence,
                    age / 10.0,
                    float(freshness is Freshness.CONTRADICTED),
                    float(freshness is Freshness.UNVERIFIED),
                )
                label = float(
                    freshness is Freshness.CONTRADICTED
                    or (
                        freshness is Freshness.UNVERIFIED
                        and age >= 6
                        and confidence <= 0.55
                    )
                )
                examples.append(DelegationExample(features=features, label=label))
    return examples
