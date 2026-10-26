"""A deterministic household environment for the mechanism MVP.

The environment deliberately exposes only observations to fast policies. The
oracle slow planner may read the hidden target location, and is labeled as such
in the UI and logs.
"""

from __future__ import annotations

from .models import Action, ActionKind, ActionResult, Observation


TARGET_OBJECT = "blue_mug"
LEFT_CABINET = "left_cabinet"
RIGHT_CABINET = "right_cabinet"
DINING_TABLE = "dining_table"
ENTRY = "entry"


class HouseholdEnvironment:
    """Small deterministic household task with an optional target-move event."""

    target_object = TARGET_OBJECT
    initial_target_location = LEFT_CABINET
    destination_location = DINING_TABLE
    locations = frozenset({ENTRY, LEFT_CABINET, RIGHT_CABINET, DINING_TABLE})

    def __init__(self, *, dynamic_event: bool = True) -> None:
        self.dynamic_event = dynamic_event
        self.reset()

    def reset(self) -> Observation:
        self.step_count = 0
        self.agent_location = ENTRY
        self.mug_location: str | None = LEFT_CABINET
        self.holding: str | None = None
        self.open_containers: set[str] = set()
        self.mug_seen_once = False
        self.event_triggered = False
        self._last_event_message: str | None = None
        return self.observe()

    def observe(self) -> Observation:
        visible: set[str] = set()
        if self.holding == TARGET_OBJECT:
            visible.add(TARGET_OBJECT)
        elif (
            self.mug_location == self.agent_location
            and self.agent_location in self.open_containers
        ):
            visible.add(TARGET_OBJECT)

        return Observation(
            step=self.step_count,
            location=self.agent_location,
            visible_objects=frozenset(visible),
            open_containers=frozenset(self.open_containers),
            holding=self.holding,
            event_message=self._last_event_message,
        )

    def step(self, action: Action) -> ActionResult:
        self._last_event_message = None
        self._maybe_move_target_before(action)
        self.step_count += 1

        if action.kind is ActionKind.NAVIGATE:
            return self._navigate(action)
        if action.kind is ActionKind.INSPECT:
            return self._inspect(action)
        if action.kind is ActionKind.PICK_UP:
            return self._pick_up(action)
        if action.kind is ActionKind.PLACE:
            return self._place(action)
        if action.kind is ActionKind.ABSTAIN:
            return self._result(True, "Agent abstained from a risky action.")
        return self._result(False, f"Unsupported environment action: {action.label()}")

    def task_complete(self) -> bool:
        return self.mug_location == DINING_TABLE and self.holding is None

    def hidden_target_location(self) -> str | None:
        """Return simulator truth; only the explicitly labeled oracle may call this."""
        return self.mug_location

    def _maybe_move_target_before(self, action: Action) -> None:
        should_move = (
            self.dynamic_event
            and not self.event_triggered
            and self.mug_seen_once
            and action.kind is ActionKind.PICK_UP
            and action.target == TARGET_OBJECT
        )
        if should_move:
            self.mug_location = RIGHT_CABINET
            self.event_triggered = True
            self._last_event_message = (
                "Dynamic event: blue_mug was moved from left_cabinet to right_cabinet."
            )

    def _navigate(self, action: Action) -> ActionResult:
        if action.target not in self.locations:
            return self._result(False, f"Unknown location: {action.target}")
        self.agent_location = action.target
        return self._result(True, f"Navigated to {action.target}.")

    def _inspect(self, action: Action) -> ActionResult:
        if action.target != self.agent_location:
            return self._result(False, f"Cannot inspect {action.target} from {self.agent_location}.")
        if action.target not in {LEFT_CABINET, RIGHT_CABINET}:
            return self._result(False, f"{action.target} is not an inspectable cabinet.")
        self.open_containers.add(action.target)
        visible = TARGET_OBJECT if self.mug_location == action.target else "no target"
        if visible == TARGET_OBJECT:
            self.mug_seen_once = True
        return self._result(True, f"Inspected {action.target}; visible: {visible}.")

    def _pick_up(self, action: Action) -> ActionResult:
        if action.target != TARGET_OBJECT:
            return self._result(False, f"Only {TARGET_OBJECT} is supported in this task.")
        if self.holding is not None:
            return self._result(False, f"Already holding {self.holding}.")
        can_pick = (
            self.mug_location == self.agent_location
            and self.agent_location in self.open_containers
        )
        if not can_pick:
            return self._result(False, f"{TARGET_OBJECT} is not available at {self.agent_location}.")
        self.holding = TARGET_OBJECT
        self.mug_location = None
        return self._result(True, f"Picked up {TARGET_OBJECT}.")

    def _place(self, action: Action) -> ActionResult:
        if action.target != DINING_TABLE:
            return self._result(False, f"This task requires placing on {DINING_TABLE}.")
        if self.agent_location != DINING_TABLE:
            return self._result(False, f"Cannot place on table from {self.agent_location}.")
        if self.holding != TARGET_OBJECT:
            return self._result(False, f"Not holding {TARGET_OBJECT}.")
        self.holding = None
        self.mug_location = DINING_TABLE
        return self._result(True, f"Placed {TARGET_OBJECT} on {DINING_TABLE}.")

    def _result(self, success: bool, message: str) -> ActionResult:
        observation = self.observe()
        if observation.event_message:
            message = f"{message} {observation.event_message}"
        return ActionResult(success=success, message=message, observation=observation)
