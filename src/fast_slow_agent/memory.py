"""Structured temporal memory with explicit freshness and contradiction states."""

from __future__ import annotations

from .environment import TARGET_OBJECT
from .models import Action, ActionKind, ActionResult, BeliefRecord, Freshness, Observation


class BeliefMemory:
    """Track one target object, defaulting to the lightweight demo's blue mug."""

    def __init__(self, *, target_object: str = TARGET_OBJECT) -> None:
        self.target_object = target_object
        self.record: BeliefRecord | None = None

    def ingest_observation(self, observation: Observation) -> None:
        # A held object is visible to the agent but is not a newly observed scene location.
        if self.target_object in observation.visible_objects and observation.holding is None:
            self.record = BeliefRecord(
                object_name=self.target_object,
                last_known_location=observation.location,
                observed_step=observation.step,
                confidence=0.92,
                freshness=Freshness.FRESH,
                source="observation",
            )

    def ingest_result(self, action: Action, result: ActionResult) -> None:
        self.ingest_observation(result.observation)
        if self.record is None:
            return

        failed_expected_pickup = (
            action.kind is ActionKind.PICK_UP
            and action.target == self.target_object
            and not result.success
            and result.observation.location == self.record.last_known_location
        )
        inspected_expected_location_without_target = (
            action.kind is ActionKind.INSPECT
            and action.target == self.record.last_known_location
            and result.success
            and self.target_object not in result.observation.visible_objects
        )
        if failed_expected_pickup or inspected_expected_location_without_target:
            self.record.confidence = 0.18
            self.record.freshness = Freshness.CONTRADICTED

    def update_from_slow_plan(self, location: str, step: int, confidence: float) -> None:
        self.record = BeliefRecord(
            object_name=self.target_object,
            last_known_location=location,
            observed_step=step,
            confidence=confidence,
            freshness=Freshness.UNVERIFIED,
            source="slow_brain_oracle",
        )

    def summary(self, current_step: int) -> str:
        if self.record is None:
            return "no target memory"
        age = current_step - self.record.observed_step
        return (
            f"{self.record.object_name} -> {self.record.last_known_location} | "
            f"confidence={self.record.confidence:.2f} | "
            f"freshness={self.record.freshness.value} | age={age} | "
            f"source={self.record.source}"
        )
