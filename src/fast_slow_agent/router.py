"""Fast action selection with a tiny learned delegation classifier."""

from __future__ import annotations

from .environment import DINING_TABLE, LEFT_CABINET, TARGET_OBJECT, HouseholdEnvironment
from .learning import TinyDelegationModel
from .memory import BeliefMemory
from .models import Action, ActionKind, Decision, Freshness


class FastRouter:
    """Fast policy plus a compact learned model for slow-model delegation.

    Ordinary manipulation/navigation actions remain deterministic in this MVP.
    The crucial ``act vs delegate`` decision is made by a fitted four-feature
    logistic classifier. It is deliberately small enough to run without GPU or
    external libraries, mirroring the role of a Jev-style fast decision layer.
    """

    def __init__(
        self,
        *,
        slow_budget: int = 2,
        delegation_model: TinyDelegationModel | None = None,
    ) -> None:
        self.slow_budget = slow_budget
        self.delegation_model = delegation_model or TinyDelegationModel.default()

    def choose(
        self,
        environment: HouseholdEnvironment,
        memory: BeliefMemory,
        slow_calls: int,
    ) -> Decision:
        if environment.holding == TARGET_OBJECT:
            if environment.agent_location != DINING_TABLE:
                return self._decision(
                    Action(ActionKind.NAVIGATE, DINING_TABLE),
                    0.98,
                    "act",
                    "Target is held; move to the destination table.",
                )
            return self._decision(
                Action(ActionKind.PLACE, DINING_TABLE),
                0.99,
                "act",
                "Target is held at the destination table.",
            )

        record = memory.record
        if record is None:
            if environment.agent_location != LEFT_CABINET:
                return self._decision(
                    Action(ActionKind.NAVIGATE, LEFT_CABINET),
                    0.72,
                    "observe",
                    "No target memory exists; inspect the initial likely cabinet.",
                )
            return self._decision(
                Action(ActionKind.INSPECT, LEFT_CABINET),
                0.74,
                "observe",
                "Need an initial observation before acting on the target.",
            )

        delegation_probability = self.delegation_model.predict_probability(
            record,
            environment.step_count,
        )
        if record.freshness is Freshness.CONTRADICTED:
            if (
                slow_calls < self.slow_budget
                and delegation_probability >= self.delegation_model.threshold
            ):
                return self._decision(
                    Action(ActionKind.DELEGATE),
                    delegation_probability,
                    "delegate",
                    "Memory conflicts with the inspected cabinet; "
                    + self.delegation_model.describe(record, environment.step_count),
                )
            return self._decision(
                Action(ActionKind.ABSTAIN),
                1.0 - delegation_probability,
                "abstain",
                "Memory is contradicted but delegation is unavailable; "
                + self.delegation_model.describe(record, environment.step_count),
            )

        location = record.last_known_location
        if environment.agent_location != location:
            confidence = 0.88 if record.freshness is Freshness.FRESH else 0.66
            return self._decision(
                Action(ActionKind.NAVIGATE, location),
                confidence,
                "act",
                f"Navigate to the last known target location: {location}.",
            )
        if location not in environment.open_containers:
            return self._decision(
                Action(ActionKind.INSPECT, location),
                0.82,
                "verify",
                "The target location has not been opened in this episode.",
            )
        return self._decision(
            Action(ActionKind.PICK_UP, TARGET_OBJECT),
            0.93 if record.freshness is Freshness.FRESH else 0.62,
            "act",
            "The remembered target cabinet is open; attempt pickup.",
        )

    @staticmethod
    def _decision(action: Action, confidence: float, mode: str, reason: str) -> Decision:
        return Decision(action=action, confidence=confidence, mode=mode, reason=reason)
