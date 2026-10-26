import unittest
from pathlib import Path

from fast_slow_agent import HouseholdEnvironment, SlowPlannerAbstained, SlowPlannerError
from fast_slow_agent.vision_planner import VisionSearchSlowPlanner
from fast_slow_agent.vision_slow_brain import (
    VisionAssessment,
    VisionCandidate,
    VisionSearchSlowBrain,
)


class FakeVisionClient:
    def __init__(self, assessment):
        self.assessment = assessment

    def assess_candidate(self, *, task, candidate):
        return self.assessment[candidate.label]


class VisionSearchSlowPlannerTests(unittest.TestCase):
    def test_converts_selected_visual_candidate_to_slow_plan(self) -> None:
        candidate = VisionCandidate("right_cabinet", (Path("frame.png"),))
        planner = VisionSearchSlowPlanner(
            task="Find the target.",
            candidate_provider=lambda environment: [candidate],
            vision_search=VisionSearchSlowBrain(
                FakeVisionClient(
                    {"right_cabinet": VisionAssessment(True, 0.85, "Target visible.")}
                )
            ),
        )

        plan = planner.resolve(HouseholdEnvironment())

        self.assertEqual(plan.target_location, "right_cabinet")
        self.assertEqual(plan.subgoal, "search_right_cabinet")
        self.assertEqual(plan.confidence, 0.85)
        self.assertEqual(plan.source, "vision_search")

    def test_raises_diagnostic_error_when_visual_search_abstains(self) -> None:
        candidate = VisionCandidate("right_cabinet", (Path("frame.png"),))
        planner = VisionSearchSlowPlanner(
            task="Find the target.",
            candidate_provider=lambda environment: [candidate],
            vision_search=VisionSearchSlowBrain(
                FakeVisionClient(
                    {"right_cabinet": VisionAssessment(False, 0.9, "Target not visible.")}
                )
            ),
        )

        with self.assertRaisesRegex(SlowPlannerAbstained, "Vision search abstained: reason=No candidate"):
            planner.resolve(HouseholdEnvironment())

    def test_wraps_empty_candidate_input_as_planner_error(self) -> None:
        planner = VisionSearchSlowPlanner(
            task="Find the target.",
            candidate_provider=lambda environment: [],
            vision_search=VisionSearchSlowBrain(FakeVisionClient({})),
        )

        with self.assertRaisesRegex(SlowPlannerError, "Vision candidate input is invalid"):
            planner.resolve(HouseholdEnvironment())


if __name__ == "__main__":
    unittest.main()
