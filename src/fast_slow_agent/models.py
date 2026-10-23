"""Shared data structures for the fast–slow embodied demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


class ActionKind(str, Enum):
    NAVIGATE = "navigate"
    INSPECT = "inspect"
    PICK_UP = "pick_up"
    PLACE = "place"
    DELEGATE = "delegate"
    ABSTAIN = "abstain"


class Freshness(str, Enum):
    FRESH = "fresh"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    target: str | None = None

    def label(self) -> str:
        return self.kind.value if self.target is None else f"{self.kind.value}({self.target})"


@dataclass(frozen=True)
class Observation:
    step: int
    location: str
    visible_objects: FrozenSet[str]
    open_containers: FrozenSet[str]
    holding: str | None
    event_message: str | None = None


@dataclass(frozen=True)
class ActionResult:
    success: bool
    message: str
    observation: Observation


@dataclass
class BeliefRecord:
    object_name: str
    last_known_location: str
    observed_step: int
    confidence: float
    freshness: Freshness
    source: str = "observation"

    @property
    def age_at_last_update(self) -> int:
        return self.observed_step


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: float
    mode: str
    reason: str


@dataclass(frozen=True)
class SlowPlan:
    target_location: str
    subgoal: str
    confidence: float
    source: str = "oracle"


@dataclass
class TraceEntry:
    step: int
    action: str
    success: bool
    message: str
    memory_summary: str
    confidence: float
    mode: str
    reason: str
    slow_calls: int


@dataclass
class EpisodeResult:
    agent_name: str
    success: bool
    failure_reason: str | None
    steps: int
    slow_calls: int
    trace: list[TraceEntry] = field(default_factory=list)
    event_triggered: bool = False
