import unittest

from fast_slow_agent import AlwaysFastAgent, FastSlowAgent, HouseholdEnvironment, run_scenario
from fast_slow_agent.environment import LEFT_CABINET, RIGHT_CABINET, TARGET_OBJECT
from fast_slow_agent.memory import BeliefMemory
from fast_slow_agent.models import Action, ActionKind, Freshness
from fast_slow_agent.router import FastRouter


class LowProbabilityDelegationModel:
    threshold = 0.5

    def predict_probability(self, record, current_step):
        return 0.1

    def describe(self, record, current_step):
        return "delegation_probability=0.10"


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
        abstain_trace = next(entry for entry in result.trace if entry.mode == "abstain")
        self.assertFalse(result.success)
        self.assertTrue(result.event_triggered)
        self.assertEqual(result.slow_calls, 0)
        self.assertIn("budget exhausted", abstain_trace.message)

    def test_unverified_slow_hint_is_inspected_before_pickup(self) -> None:
        environment = HouseholdEnvironment(dynamic_event=False)
        environment.step(Action(ActionKind.NAVIGATE, RIGHT_CABINET))
        memory = BeliefMemory()
        memory.update_from_slow_plan(RIGHT_CABINET, step=0, confidence=0.8)
        router = FastRouter()

        decision = router.choose(environment, memory, slow_calls=0)
        self.assertEqual(decision.action, Action(ActionKind.INSPECT, RIGHT_CABINET))
        self.assertEqual(decision.mode, "verify")

        result = environment.step(decision.action)
        memory.ingest_result(decision.action, result)
        self.assertIsNotNone(memory.record)
        self.assertEqual(memory.record.freshness, Freshness.CONTRADICTED)

    def test_abstain_reason_distinguishes_low_delegation_probability(self) -> None:
        environment = HouseholdEnvironment(dynamic_event=True)
        memory = BeliefMemory()
        environment.step(Action(ActionKind.NAVIGATE, LEFT_CABINET))
        inspected = environment.step(Action(ActionKind.INSPECT, LEFT_CABINET))
        memory.ingest_result(Action(ActionKind.INSPECT, LEFT_CABINET), inspected)
        failed_pickup = environment.step(Action(ActionKind.PICK_UP, TARGET_OBJECT))
        memory.ingest_result(Action(ActionKind.PICK_UP, TARGET_OBJECT), failed_pickup)

        router = FastRouter(delegation_model=LowProbabilityDelegationModel())
        decision = router.choose(environment, memory, slow_calls=0)
        self.assertEqual(decision.action.kind, ActionKind.ABSTAIN)
        self.assertIn("below the threshold", decision.reason)

    def test_rejects_negative_slow_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            FastSlowAgent(slow_budget=-1)

    def test_rejects_non_positive_step_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "step budget must be positive"):
            FastSlowAgent().run(HouseholdEnvironment(), max_steps=0)


if __name__ == "__main__":
    unittest.main()
