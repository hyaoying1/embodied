"""Adapter from AI2-THOR metadata to the shared structured memory pipeline."""

from __future__ import annotations

from typing import Any

from .memory import BeliefMemory
from .models import Action, ActionKind, ActionResult, Observation


class ThorMemoryBridge:
    """Convert real THOR observation/action results into ``BeliefMemory`` updates."""

    def __init__(self, *, target_object_id: str) -> None:
        self.target_object_id = target_object_id
        self.memory = BeliefMemory(target_object=target_object_id)

    def ingest_observation(self, event: Any, *, location: str, step: int) -> Observation:
        observation = self._observation_from_event(event, location=location, step=step)
        self.memory.ingest_observation(observation)
        return observation

    def ingest_pickup_result(self, event: Any, *, location: str, step: int) -> ActionResult:
        observation = self._observation_from_event(event, location=location, step=step)
        result = ActionResult(
            success=bool(event.metadata["lastActionSuccess"]),
            message=event.metadata.get("errorMessage", ""),
            observation=observation,
        )
        self.memory.ingest_result(Action(ActionKind.PICK_UP, self.target_object_id), result)
        return result

    def _observation_from_event(self, event: Any, *, location: str, step: int) -> Observation:
        target = next(
            obj for obj in event.metadata["objects"] if obj["objectId"] == self.target_object_id
        )
        visible_objects = frozenset({self.target_object_id}) if target.get("visible") else frozenset()
        return Observation(
            step=step,
            location=location,
            visible_objects=visible_objects,
            open_containers=frozenset(),
            holding=None,
            event_message=event.metadata.get("errorMessage") or None,
        )
