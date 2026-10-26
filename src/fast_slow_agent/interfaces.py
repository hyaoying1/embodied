"""Shared contracts for fast-slow control-loop backends."""

from __future__ import annotations

from collections.abc import Set
from typing import Protocol, runtime_checkable

from .models import Action, ActionResult, SlowPlan


@runtime_checkable
class AgentEnvironment(Protocol):
    """Minimal environment surface consumed by the current agent workflow.

    ``hidden_target_location`` is intentionally included for the temporary
    oracle Slow Brain. A production visual slow brain should not require it.
    """

    target_object: str
    initial_target_location: str
    destination_location: str
    step_count: int
    agent_location: str
    holding: str | None
    open_containers: Set[str]
    event_triggered: bool

    def step(self, action: Action) -> ActionResult:
        """Execute one high-level action and return the resulting observation."""

    def task_complete(self) -> bool:
        """Return whether the environment's task objective is complete."""

    def hidden_target_location(self) -> str | None:
        """Return simulator truth for the explicitly labeled oracle planner."""


class SlowPlannerError(RuntimeError):
    """A recoverable domain failure from a delegated slow planner."""


class SlowPlannerAbstained(SlowPlannerError):
    """A slow planner intentionally declined to provide a subgoal."""


@runtime_checkable
class SlowPlanner(Protocol):
    """A replaceable slow reasoning layer for the shared agent loop."""

    name: str

    def resolve(self, environment: AgentEnvironment) -> SlowPlan:
        """Return a location subgoal after the Router delegates."""
