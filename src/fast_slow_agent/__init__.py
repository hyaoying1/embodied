"""Fast–slow embodied delegation MVP."""

from .agent import AlwaysFastAgent, FastSlowAgent
from .environment import HouseholdEnvironment
from .evaluation import run_scenario

__all__ = ["AlwaysFastAgent", "FastSlowAgent", "HouseholdEnvironment", "run_scenario"]
