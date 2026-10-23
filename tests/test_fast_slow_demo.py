import unittest

from fast_slow_agent import AlwaysFastAgent, FastSlowAgent, HouseholdEnvironment, run_scenario
from fast_slow_agent.environment import LEFT_CABINET, RIGHT_CABINET, TARGET_OBJECT
from fast_slow_agent.models import Action, ActionKind


class FastSlowDemoTests(unittest.TestCase):
    def test_dynamic_event_moves_target_after_initial_observation(self) -> None:
        environment = HouseholdEnvironment(dynamic_event=True)
        environment.step(Action(ActionKind.NAVIGATE, LEFT_CABINET))
        inspected = environment.step(Action(ActionKind.INSPECT, LEFT_CABINET))
        self.assertIn(TARGET_OBJECT, inspected.observation.visible_objects)

        failed_pickup = environment.step(Action(ActionKind.PICK_UP, TARGET_OBJECT))
        self.assertFalse(failed_pickup.success)
        self.assertTrue(environment.event_triggered)
        self.assertEqual(environment.hidden_target_location(), RIGHT_CABINET)

    def test_always_fast_fails_after_target_move(self) -> None:
        result = run_scenario(AlwaysFastAgent(), dynamic_event=True)
        self.assertFalse(result.success)
        self.assertTrue(result.event_triggered)
        self.assertEqual(result.slow_calls, 0)

    def test_fast_slow_delegates_and_recovers_after_target_move(self) -> None:
        result = run_scenario(FastSlowAgent(), dynamic_event=True)
        self.assertTrue(result.success)
        self.assertTrue(result.event_triggered)
        self.assertEqual(result.slow_calls, 1)
        self.assertTrue(any(entry.mode == "delegate" for entry in result.trace))

    def test_fast_slow_does_not_delegate_when_world_is_stable(self) -> None:
        result = run_scenario(FastSlowAgent(), dynamic_event=False)
        self.assertTrue(result.success)
        self.assertFalse(result.event_triggered)
        self.assertEqual(result.slow_calls, 0)
    def test_fast_slow_abstains_when_conflict_has_no_slow_budget(self) -> None:
        result = run_scenario(FastSlowAgent(slow_budget=0), dynamic_event=True)
        self.assertFalse(result.success)
        self.assertTrue(result.event_triggered)
        self.assertEqual(result.slow_calls, 0)
        self.assertTrue(any(entry.mode == "abstain" for entry in result.trace))


if __name__ == "__main__":
    unittest.main()
