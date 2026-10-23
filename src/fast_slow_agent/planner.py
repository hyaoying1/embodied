"""Slow planning layer.

The MVP intentionally uses an oracle implementation to validate the delegation
control flow. It is not presented as a real VLM or LLM capability.
"""

from __future__ import annotations

from .environment import DINING_TABLE, HouseholdEnvironment
from .models import SlowPlan


class OracleSlowPlanner:
    name = "oracle_slow_planner"

    def resolve(self, environment: HouseholdEnvironment) -> SlowPlan:
        location = environment.hidden_target_location()
        if location is None:
            return SlowPlan(
                target_location=DINING_TABLE,
                subgoal="place_held_mug_on_table",
                confidence=0.99,
            )
        return SlowPlan(
            target_location=location,
            subgoal=f"search_{location}",
            confidence=0.78,
        )
