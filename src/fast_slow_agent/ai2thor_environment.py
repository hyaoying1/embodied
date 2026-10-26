"""AI2-THOR implementation of the shared AgentEnvironment contract.

This adapter intentionally maps high-level actions to AI2-THOR's simulator
bridges (interactable poses and teleportation). It is suitable for controlled
3D evaluation, not a learned navigation policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import Action, ActionKind, ActionResult, Observation


class AI2ThorEnvironment:
    """Map high-level fast-slow actions onto an existing AI2-THOR controller.

    ``locations`` maps stable agent-facing labels to AI2-THOR receptacle IDs.
    The adapter treats a successful ``INSPECT`` as an opened or verified area;
    this preserves the current Router's inspection gate for openable cabinets
    and open-surface counters alike.
    """

    def __init__(
        self,
        *,
        controller: Any,
        target_object_id: str,
        initial_target_location: str,
        destination_location: str,
        locations: Mapping[str, str],
        navigation_object_ids: Mapping[str, str] | None = None,
        dynamic_event: bool = False,
        moved_target_location: str | None = None,
    ) -> None:
        if initial_target_location not in locations or destination_location not in locations:
            raise ValueError("Initial and destination locations must be mapped THOR receptacles.")
        if dynamic_event and (moved_target_location is None or moved_target_location not in locations):
            raise ValueError("Dynamic THOR events require a mapped moved target location.")
        self.controller = controller
        self.target_object = target_object_id
        self.initial_target_location = initial_target_location
        self.destination_location = destination_location
        self.locations = dict(locations)
        self.navigation_object_ids = dict(navigation_object_ids or locations)
        if set(self.navigation_object_ids) != set(self.locations):
            raise ValueError("Navigation object IDs must be provided for exactly the mapped locations.")
        self.dynamic_event = dynamic_event
        self.moved_target_location = moved_target_location
        self._location_by_object_id = {object_id: label for label, object_id in locations.items()}
        self.step_count = 0
        self.agent_location = ""
        self.holding: str | None = None
        self.open_containers: set[str] = set()
        self.event_triggered = False
        self._placed_successfully = False
        self._target_seen_once = False
        self._event_message: str | None = None

    def observe(self) -> Observation:
        return self._observation_from_event(self.controller.last_event)

    def step(self, action: Action) -> ActionResult:
        self.step_count += 1
        self._event_message = None
        if action.kind is ActionKind.NAVIGATE:
            return self._navigate(action)
        if action.kind is ActionKind.INSPECT:
            return self._inspect(action)
        if action.kind is ActionKind.PICK_UP:
            if self._should_move_target(action):
                move_error = self._move_target_for_dynamic_event()
                if move_error is not None:
                    return self._result(False, move_error)
            return self._pick_up(action)
        if action.kind is ActionKind.PLACE:
            return self._place(action)
        if action.kind is ActionKind.ABSTAIN:
            return self._result(True, "Agent abstained from a risky action.")
        return self._result(False, f"Unsupported environment action: {action.label()}")

    def task_complete(self) -> bool:
        return self._placed_successfully and self.holding is None

    def hidden_target_location(self) -> str | None:
        """Return simulator truth solely for the explicitly labeled oracle planner."""
        target = self._target_metadata(self.controller.last_event)
        if target is None or self.holding == self.target_object:
            return None
        for receptacle_id in target.get("parentReceptacles") or ():
            label = self._location_by_object_id.get(receptacle_id)
            if label is not None:
                return label
        return None

    def _should_move_target(self, action: Action) -> bool:
        return (
            self.dynamic_event
            and not self.event_triggered
            and self._target_seen_once
            and action.target == self.target_object
        )

    def _move_target_for_dynamic_event(self) -> str | None:
        assert self.moved_target_location is not None
        destination_id = self.locations[self.moved_target_location]
        spawn_event = self.controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=destination_id,
            anywhere=True,
        )
        if not self._succeeded(spawn_event):
            return self._failure_message(spawn_event, "Could not prepare dynamic target move")
        points = spawn_event.metadata.get("actionReturn") or []
        if not points:
            return f"No clear spawn point found for dynamic target location {self.moved_target_location}."
        move_event = self.controller.step(
            action="PlaceObjectAtPoint",
            objectId=self.target_object,
            positions=points,
            forceKinematic=True,
            raise_for_failure=False,
        )
        if not self._succeeded(move_event):
            return self._failure_message(move_event, "Could not move target for dynamic event")
        self.event_triggered = True
        self._event_message = (
            f"Dynamic event: {self.target_object} moved to {self.moved_target_location}."
        )
        return None

    def _navigate(self, action: Action) -> ActionResult:
        if action.target not in self.locations:
            return self._result(False, f"Unknown THOR location: {action.target}")
        object_id = self.navigation_object_ids[action.target]
        poses_event = self.controller.step(
            action="GetInteractablePoses",
            objectId=object_id,
            rotations=[0, 45, 90, 135, 180, 225, 270, 315],
            horizons=[0],
            standings=[True],
        )
        if not self._succeeded(poses_event):
            return self._result(False, self._failure_message(poses_event, "Could not query interactable poses"))
        poses = poses_event.metadata.get("actionReturn") or []
        if not poses:
            return self._result(False, f"No interactable pose found for {action.target}.")
        pose = poses[0]
        event = self.controller.step(
            action="TeleportFull",
            position={"x": pose["x"], "y": pose["y"], "z": pose["z"]},
            rotation={"x": 0, "y": pose["rotation"], "z": 0},
            horizon=pose["horizon"],
            standing=pose["standing"],
        )
        if not self._succeeded(event):
            return self._result(False, self._failure_message(event, "Could not teleport"))
        self.agent_location = action.target
        return self._result(True, f"Navigated to {action.target}.", event)

    def _inspect(self, action: Action) -> ActionResult:
        if action.target != self.agent_location:
            return self._result(False, f"Cannot inspect {action.target} from {self.agent_location}.")
        object_id = self.locations.get(action.target or "")
        if object_id is None:
            return self._result(False, f"Unknown THOR location: {action.target}")
        receptacle = self._object_metadata(self.controller.last_event, object_id)
        if receptacle and receptacle.get("openable"):
            event = self.controller.step(action="OpenObject", objectId=object_id, raise_for_failure=False)
            if not self._succeeded(event):
                return self._result(False, self._failure_message(event, f"Could not inspect {action.target}"))
        else:
            event = self.controller.last_event
        self.open_containers.add(action.target)
        return self._result(True, f"Inspected {action.target}.", event)

    def _pick_up(self, action: Action) -> ActionResult:
        if action.target != self.target_object:
            return self._result(False, f"Only configured target {self.target_object} can be picked up.")
        if self.holding is not None:
            return self._result(False, f"Already holding {self.holding}.")
        if self.agent_location not in self.open_containers:
            return self._result(False, f"Target area {self.agent_location} has not been inspected.")
        event = self.controller.step(action="PickupObject", objectId=self.target_object, raise_for_failure=False)
        if self._succeeded(event):
            self.holding = self.target_object
            return self._result(True, f"Picked up {self.target_object}.", event)
        return self._result(False, self._failure_message(event, f"Could not pick up {self.target_object}"), event)

    def _place(self, action: Action) -> ActionResult:
        if action.target != self.destination_location:
            return self._result(False, f"This task requires placing on {self.destination_location}.")
        if self.agent_location != self.destination_location:
            return self._result(False, f"Cannot place at {action.target} from {self.agent_location}.")
        if self.holding != self.target_object:
            return self._result(False, f"Not holding {self.target_object}.")
        event = self.controller.step(
            action="PutObject",
            objectId=self.locations[self.destination_location],
            raise_for_failure=False,
        )
        if not self._succeeded(event):
            return self._result(False, self._failure_message(event, f"Could not place {self.target_object}"), event)
        self.holding = None
        self._placed_successfully = True
        return self._result(True, f"Placed {self.target_object} on {self.destination_location}.", event)

    def _result(self, success: bool, message: str, event: Any | None = None) -> ActionResult:
        observation = self._observation_from_event(event or self.controller.last_event)
        if observation.event_message:
            message = f"{message} {observation.event_message}"
        return ActionResult(success=success, message=message, observation=observation)

    def _observation_from_event(self, event: Any) -> Observation:
        target = self._target_metadata(event)
        visible = frozenset({self.target_object}) if target and target.get("visible") else frozenset()
        if visible and self.holding is None:
            self._target_seen_once = True
        event_message = self._event_message or event.metadata.get("errorMessage") or None
        return Observation(
            step=self.step_count,
            location=self.agent_location,
            visible_objects=visible,
            open_containers=frozenset(self.open_containers),
            holding=self.holding,
            event_message=event_message,
        )

    def _target_metadata(self, event: Any) -> dict[str, Any] | None:
        return self._object_metadata(event, self.target_object)

    @staticmethod
    def _object_metadata(event: Any, object_id: str) -> dict[str, Any] | None:
        return next(
            (obj for obj in event.metadata.get("objects", ()) if obj.get("objectId") == object_id),
            None,
        )

    @staticmethod
    def _succeeded(event: Any) -> bool:
        return bool(event.metadata.get("lastActionSuccess", False))

    @staticmethod
    def _failure_message(event: Any, prefix: str) -> str:
        return f"{prefix}: {event.metadata.get('errorMessage', 'unknown AI2-THOR error')}"
