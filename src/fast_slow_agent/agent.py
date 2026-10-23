"""Always-fast and fast–slow agents sharing a deterministic control loop."""

from __future__ import annotations

from .environment import DINING_TABLE, LEFT_CABINET, TARGET_OBJECT, HouseholdEnvironment
from .memory import BeliefMemory
from .models import Action, ActionKind, Decision, EpisodeResult, TraceEntry
from .planner import OracleSlowPlanner
from .router import FastRouter


class BaseAgent:
    name = "base"

    def run(self, environment: HouseholdEnvironment, *, max_steps: int = 12) -> EpisodeResult:
        raise NotImplementedError

    @staticmethod
    def _trace(
        trace: list[TraceEntry],
        environment: HouseholdEnvironment,
        decision: Decision,
        *,
        success: bool,
        message: str,
        memory: BeliefMemory,
        slow_calls: int,
    ) -> None:
        trace.append(
            TraceEntry(
                step=environment.step_count,
                action=decision.action.label(),
                success=success,
                message=message,
                memory_summary=memory.summary(environment.step_count),
                confidence=decision.confidence,
                mode=decision.mode,
                reason=decision.reason,
                slow_calls=slow_calls,
            )
        )


class AlwaysFastAgent(BaseAgent):
    """A baseline that always trusts its first target memory."""

    name = "Always Fast"

    def run(self, environment: HouseholdEnvironment, *, max_steps: int = 12) -> EpisodeResult:
        memory = BeliefMemory()
        trace: list[TraceEntry] = []
        for _ in range(max_steps):
            decision = self._choose(environment, memory)
            result = environment.step(decision.action)
            memory.ingest_result(decision.action, result)
            self._trace(
                trace,
                environment,
                decision,
                success=result.success,
                message=result.message,
                memory=memory,
                slow_calls=0,
            )
            if environment.task_complete():
                return EpisodeResult(
                    agent_name=self.name,
                    success=True,
                    failure_reason=None,
                    steps=environment.step_count,
                    slow_calls=0,
                    trace=trace,
                    event_triggered=environment.event_triggered,
                )
        return EpisodeResult(
            agent_name=self.name,
            success=False,
            failure_reason="Maximum steps reached while repeating stale-memory behavior.",
            steps=environment.step_count,
            slow_calls=0,
            trace=trace,
            event_triggered=environment.event_triggered,
        )

    def _choose(self, environment: HouseholdEnvironment, memory: BeliefMemory) -> Decision:
        if environment.holding == TARGET_OBJECT:
            if environment.agent_location != DINING_TABLE:
                return Decision(Action(ActionKind.NAVIGATE, DINING_TABLE), 0.98, "act", "Target held.")
            return Decision(Action(ActionKind.PLACE, DINING_TABLE), 0.99, "act", "Place target.")
        if memory.record is None:
            if environment.agent_location != LEFT_CABINET:
                return Decision(Action(ActionKind.NAVIGATE, LEFT_CABINET), 0.72, "act", "Initial search.")
            return Decision(Action(ActionKind.INSPECT, LEFT_CABINET), 0.74, "act", "Inspect initial cabinet.")
        location = memory.record.last_known_location
        if environment.agent_location != location:
            return Decision(Action(ActionKind.NAVIGATE, location), 0.94, "act", "Trust last known location.")
        if location not in environment.open_containers:
            return Decision(Action(ActionKind.INSPECT, location), 0.94, "act", "Trust last known location.")
        return Decision(Action(ActionKind.PICK_UP, TARGET_OBJECT), 0.96, "act", "Trust stale memory and retry pickup.")


class FastSlowAgent(BaseAgent):
    name = "Fast–Slow Router"

    def __init__(self, *, slow_budget: int = 2) -> None:
        self.router = FastRouter(slow_budget=slow_budget)
        self.slow_planner = OracleSlowPlanner()

    def run(self, environment: HouseholdEnvironment, *, max_steps: int = 12) -> EpisodeResult:
        memory = BeliefMemory()
        trace: list[TraceEntry] = []
        slow_calls = 0

        for _ in range(max_steps):
            decision = self.router.choose(environment, memory, slow_calls)
            if decision.action.kind is ActionKind.DELEGATE:
                slow_calls += 1
                plan = self.slow_planner.resolve(environment)
                memory.update_from_slow_plan(
                    plan.target_location,
                    environment.step_count,
                    plan.confidence,
                )
                self._trace(
                    trace,
                    environment,
                    decision,
                    success=True,
                    message=(
                        f"Slow Brain ({self.slow_planner.name}) returned "
                        f"subgoal={plan.subgoal}, target_location={plan.target_location}."
                    ),
                    memory=memory,
                    slow_calls=slow_calls,
                )
                continue
            if decision.action.kind is ActionKind.ABSTAIN:
                self._trace(
                    trace,
                    environment,
                    decision,
                    success=True,
                    message="Agent abstained because the slow-model budget is exhausted.",
                    memory=memory,
                    slow_calls=slow_calls,
                )
                break

            result = environment.step(decision.action)
            memory.ingest_result(decision.action, result)
            self._trace(
                trace,
                environment,
                decision,
                success=result.success,
                message=result.message,
                memory=memory,
                slow_calls=slow_calls,
            )
            if environment.task_complete():
                return EpisodeResult(
                    agent_name=self.name,
                    success=True,
                    failure_reason=None,
                    steps=environment.step_count,
                    slow_calls=slow_calls,
                    trace=trace,
                    event_triggered=environment.event_triggered,
                )

        return EpisodeResult(
            agent_name=self.name,
            success=False,
            failure_reason="Task was not completed before the step or delegation budget ended.",
            steps=environment.step_count,
            slow_calls=slow_calls,
            trace=trace,
            event_triggered=environment.event_triggered,
        )
