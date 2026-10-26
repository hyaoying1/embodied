"""Fast action selection with a tiny learned delegation classifier."""

from __future__ import annotations

from .environment import DINING_TABLE, LEFT_CABINET, TARGET_OBJECT, HouseholdEnvironment
from .learning import TinyDelegationModel
from .memory import BeliefMemory
from .models import Action, ActionKind, BeliefRecord, Decision, Freshness


class FastRouter:
    """Deterministic fast policy with learned escalation for contradicted memory."""

    def __init__(
        self,
        *,
        slow_budget: int = 2,
        delegation_model: TinyDelegationModel | None = None,
    ) -> None:
        if slow_budget < 0:
            raise ValueError("Slow Brain budget cannot be negative.")
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
            return self._delegate_or_abstain(
                record=record,
                delegation_probability=delegation_probability,
                current_step=environment.step_count,
                slow_calls=slow_calls,
                reason_prefix="Memory conflicts with the latest action outcome.",
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
        if record.freshness is Freshness.UNVERIFIED:
            return self._decision(
                Action(ActionKind.INSPECT, location),
                0.72,
                "verify",
                "Slow Brain location hint must be verified before attempting pickup.",
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
            0.93,
            "act",
            "The remembered target cabinet is open; attempt pickup.",
        )

    def _delegate_or_abstain(
        self,
        *,
        record: BeliefRecord,
        delegation_probability: float,
        current_step: int,
        slow_calls: int,
        reason_prefix: str,
    ) -> Decision:
        model_detail = self.delegation_model.describe(record, current_step)
        if slow_calls >= self.slow_budget:
            return self._decision(
                Action(ActionKind.ABSTAIN),
                1.0 - delegation_probability,
                "abstain",
                f"{reason_prefix} Slow Brain budget exhausted; {model_detail}",
            )
        if delegation_probability < self.delegation_model.threshold:
            return self._decision(
                Action(ActionKind.ABSTAIN),
                1.0 - delegation_probability,
                "abstain",
                f"{reason_prefix} Delegation probability is below the threshold; {model_detail}",
            )
        return self._decision(
            Action(ActionKind.DELEGATE),
            delegation_probability,
            "delegate",
            f"{reason_prefix} {model_detail}",
        )

    @staticmethod
    def _decision(action: Action, confidence: float, mode: str, reason: str) -> Decision:
        return Decision(action=action, confidence=confidence, mode=mode, reason=reason)
