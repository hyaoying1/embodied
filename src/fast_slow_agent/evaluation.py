"""Scenario runner and small comparison helpers."""

from __future__ import annotations

from .agent import AlwaysFastAgent, BaseAgent, FastSlowAgent
from .environment import HouseholdEnvironment
from .models import EpisodeResult


def run_scenario(agent: BaseAgent, *, dynamic_event: bool) -> EpisodeResult:
    environment = HouseholdEnvironment(dynamic_event=dynamic_event)
    return agent.run(environment)


def comparison_suite() -> list[tuple[str, EpisodeResult]]:
    scenarios: list[tuple[str, BaseAgent, bool]] = [
        ("Always Fast with target move", AlwaysFastAgent(), True),
        ("Fast–Slow Router with target move", FastSlowAgent(), True),
        ("Fast–Slow Router without target move", FastSlowAgent(), False),
        ("Fast–Slow Router with no slow-model budget", FastSlowAgent(slow_budget=0), True),
    ]
    return [
        (label, run_scenario(agent, dynamic_event=dynamic_event))
        for label, agent, dynamic_event in scenarios
    ]
