"""Always-fast and fast–slow agents sharing a deterministic control loop."""

from __future__ import annotations

from .interfaces import AgentEnvironment, SlowPlanner
from .memory import BeliefMemory
from .models import Action, ActionKind, Decision, EpisodeResult, TraceEntry
from .planner import OracleSlowPlanner
from .router import FastRouter


class BaseAgent:
    name = "base"

    def run(self, environment: AgentEnvironment, *, max_steps: int = 12) -> EpisodeResult:
        raise NotImplementedError

    @staticmethod
    def _trace(
        trace: list[TraceEntry],
        environment: AgentEnvironment,
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

    def run(self, environment: AgentEnvironment, *, max_steps: int = 12) -> EpisodeResult:
        if max_steps <= 0:
            raise ValueError("Maximum step budget must be positive.")
        memory = BeliefMemory(target_object=environment.target_object)
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

    def _choose(self, environment: AgentEnvironment, memory: BeliefMemory) -> Decision:
        if environment.holding == environment.target_object:
            if environment.agent_location != environment.destination_location:
                return Decision(Action(ActionKind.NAVIGATE, environment.destination_location), 0.98, "act", "Target held.")
            return Decision(Action(ActionKind.PLACE, environment.destination_location), 0.99, "act", "Place target.")
        if memory.record is None:
            if environment.agent_location != environment.initial_target_location:
                return Decision(Action(ActionKind.NAVIGATE, environment.initial_target_location), 0.72, "act", "Initial search.")
            return Decision(Action(ActionKind.INSPECT, environment.initial_target_location), 0.74, "act", "Inspect initial cabinet.")
        location = memory.record.last_known_location
        if environment.agent_location != location:
            return Decision(Action(ActionKind.NAVIGATE, location), 0.94, "act", "Trust last known location.")
        if location not in environment.open_containers:
            return Decision(Action(ActionKind.INSPECT, location), 0.94, "act", "Trust last known location.")
        return Decision(Action(ActionKind.PICK_UP, environment.target_object), 0.96, "act", "Trust stale memory and retry pickup.")


class FastSlowAgent(BaseAgent):
    name = "Fast–Slow Router"

    def __init__(
        self,
        *,
        slow_budget: int = 2,
        slow_planner: SlowPlanner | None = None,
    ) -> None:
        self.router = FastRouter(slow_budget=slow_budget)
        self.slow_planner = slow_planner or OracleSlowPlanner()

    def run(self, environment: AgentEnvironment, *, max_steps: int = 12) -> EpisodeResult:
        if max_steps <= 0:
            raise ValueError("Maximum step budget must be positive.")
        memory = BeliefMemory(target_object=environment.target_object)
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
                    message=decision.reason,
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
