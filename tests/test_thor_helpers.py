from types import SimpleNamespace
import unittest

from fast_slow_agent.thor_demo import AI2ThorFastSlowDemo


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


if __name__ == "__main__":
    unittest.main()
