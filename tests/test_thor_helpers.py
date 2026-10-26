from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fast_slow_agent.thor_demo import AI2ThorFastSlowDemo, ThorDemoResult
from fast_slow_agent.vision_slow_brain import VisionSearchResult


class FakeController:
    def __init__(self, *events) -> None:
        self.events = list(events)
        self.calls: list[dict[str, object]] = []

    def step(self, **kwargs):
        self.calls.append(kwargs)
        return self.events.pop(0)


def event(*, success: bool, action_return=None, error: str = "simulator failure"):
    return SimpleNamespace(
        metadata={
            "lastActionSuccess": success,
            "actionReturn": action_return,
            "errorMessage": error,
        }
    )


class AbstainingVisionSlowBrain:
    def locate(self, *, task, candidates) -> VisionSearchResult:
        return VisionSearchResult(
            None,
            None,
            None,
            {},
            len(candidates),
            0,
            {"counter_a": "request timed out"},
            "No candidate produced a positive visual assessment.",
        )


class ThorHelperTests(unittest.TestCase):
    def test_evenly_samples_requested_number_of_search_poses(self) -> None:
        poses = [{"index": index} for index in range(20)]
        sampled = AI2ThorFastSlowDemo._sample_search_poses(poses, max_views=8)
        self.assertEqual(len(sampled), 8)
        self.assertEqual(sampled[0]["index"], 0)
        self.assertEqual(sampled[-1]["index"], 19)

    def test_keeps_all_poses_when_below_view_budget(self) -> None:
        poses = [{"index": index} for index in range(3)]
        self.assertEqual(AI2ThorFastSlowDemo._sample_search_poses(poses, max_views=8), poses)

    def test_rejects_non_positive_view_budget(self) -> None:
        with self.assertRaises(ValueError):
            AI2ThorFastSlowDemo._sample_search_poses([{"index": 1}], max_views=0)

    def test_interactable_pose_failure_includes_simulator_message(self) -> None:
        controller = FakeController(event(success=False, error="object is unreachable"))
        with self.assertRaisesRegex(RuntimeError, "Could not query interactable poses: object is unreachable"):
            AI2ThorFastSlowDemo._interactable_poses(controller, "Mug|1")

    def test_teleport_failure_includes_simulator_message(self) -> None:
        controller = FakeController(event(success=False, error="blocked position"))
        pose = {"x": 0, "y": 0, "z": 0, "rotation": 0, "horizon": 0, "standing": True}
        with self.assertRaisesRegex(RuntimeError, "Could not teleport: blocked position"):
            AI2ThorFastSlowDemo._teleport(controller, pose)

    def test_spawn_query_failure_includes_simulator_message(self) -> None:
        controller = FakeController(event(success=False, error="receptacle is full"))
        with self.assertRaisesRegex(RuntimeError, "Could not query receptacle spawn points: receptacle is full"):
            AI2ThorFastSlowDemo._move_mug_to_receptacle(controller, "Mug|1", "CounterTop|1")

    def test_move_failure_includes_simulator_message(self) -> None:
        controller = FakeController(
            event(success=True, action_return=[{"x": 0, "y": 0, "z": 0}]),
            event(success=False, error="mug is obstructed"),
        )
        with self.assertRaisesRegex(RuntimeError, "Could not move mug: mug is obstructed"):
            AI2ThorFastSlowDemo._move_mug_to_receptacle(controller, "Mug|1", "CounterTop|1")

    def test_vlm_abstain_returns_diagnostic_instead_of_raising(self) -> None:
        demo = AI2ThorFastSlowDemo(
            slow_brain_mode="vlm",
            vision_slow_brain=AbstainingVisionSlowBrain(),
        )
        counters = [{"objectId": "counter_a"}, {"objectId": "counter_b"}]
        with patch.object(demo, "_capture_candidate_views", return_value=([], [])):
            selected, detail, frames = demo._select_recovery_counter(
                object(),
                counters=counters,
                oracle_moved_counter=counters[0],
            )
        self.assertIsNone(selected)
        self.assertEqual(frames, [])
        self.assertIn("VLM recovery search abstained", detail)
        self.assertIn("request timed out", detail)

    def test_failed_thor_result_is_not_successful(self) -> None:
        result = ThorDemoResult(
            scene="FloorPlan1",
            source_counter_id="counter_a",
            moved_counter_id="counter_b",
            selected_counter_id=None,
            destination_counter_id="counter_c",
            slow_brain_mode="vlm",
            slow_brain_detail="VLM recovery search abstained.",
            delegate_probability=0.9,
            source_pickup_failed=True,
            recovered_pickup_succeeded=False,
            placement_succeeded=False,
            frames=(),
            failure_reason="VLM recovery search abstained.",
        )
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
