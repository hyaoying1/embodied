"""Bridge bounded vision candidate search into the shared SlowPlanner contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .interfaces import AgentEnvironment, SlowPlannerAbstained, SlowPlannerError
from .models import SlowPlan
from .vision_slow_brain import VisionCandidate, VisionSearchSlowBrain


VisionCandidateProvider = Callable[[AgentEnvironment], Sequence[VisionCandidate]]


class VisionSearchSlowPlanner:
    """Use a bounded visual search result as a delegated location subgoal."""

    name = "vision_search_slow_planner"

    def __init__(
        self,
        *,
        task: str,
        candidate_provider: VisionCandidateProvider,
        vision_search: VisionSearchSlowBrain,
    ) -> None:
        if not task:
            raise ValueError("Vision planner task must not be empty.")
        self.task = task
        self.candidate_provider = candidate_provider
        self.vision_search = vision_search

    def resolve(self, environment: AgentEnvironment) -> SlowPlan:
        try:
            candidates = self.candidate_provider(environment)
            result = self.vision_search.locate(task=self.task, candidates=candidates)
        except ValueError as error:
            raise SlowPlannerError(f"Vision candidate input is invalid: {error}") from error
        if result.selected_label is None:
            errors = "; ".join(
                f"{label}: {message}" for label, message in result.candidate_errors.items()
            ) or "none"
            raise SlowPlannerAbstained(
                "Vision search abstained: "
                f"reason={result.abstain_reason}; candidate_errors={errors}"
            )
        assert result.confidence is not None
        return SlowPlan(
            target_location=result.selected_label,
            subgoal=f"search_{result.selected_label}",
            confidence=result.confidence,
            source="vision_search",
        )
