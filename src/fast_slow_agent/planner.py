"""Slow planning layer.

The MVP intentionally uses an oracle implementation to validate the delegation
control flow. It is not presented as a real VLM or LLM capability.
"""

from __future__ import annotations

from .interfaces import AgentEnvironment
from .models import SlowPlan


class OracleSlowPlanner:
    name = "oracle_slow_planner"

    def resolve(self, environment: AgentEnvironment) -> SlowPlan:
        location = environment.hidden_target_location()
        if location is None:
            return SlowPlan(
                target_location=environment.destination_location,
                subgoal="place_held_mug_on_table",
                confidence=0.99,
            )
        return SlowPlan(
            target_location=location,
            subgoal=f"search_{location}",
            confidence=0.78,
        )
