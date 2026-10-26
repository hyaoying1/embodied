from types import SimpleNamespace
import unittest

from fast_slow_agent.agent import FastSlowAgent
from fast_slow_agent.ai2thor_environment import AI2ThorEnvironment
from fast_slow_agent.interfaces import AgentEnvironment
from fast_slow_agent.memory import BeliefMemory
from fast_slow_agent.models import Action, ActionKind, Freshness


TARGET_ID = "Mug|1"
SOURCE_ID = "CounterTop|source"
MOVED_ID = "CounterTop|moved"
DESTINATION_ID = "CounterTop|destination"
POSE = {"x": 1.0, "y": 0.9, "z": 2.0, "rotation": 90, "horizon": 0, "standing": True}


def thor_event(*, success=True, objects=(), action_return=None, error=""):
    return SimpleNamespace(
        metadata={
            "lastActionSuccess": success,
            "objects": list(objects),
            "actionReturn": action_return,
            "errorMessage": error,
        }
    )


def target(*, visible, parents=()):
    return {"objectId": TARGET_ID, "visible": visible, "parentReceptacles": list(parents)}


class ScriptedController:
    def __init__(self, initial_event, *events) -> None:
        self.last_event = initial_event
        self.events = list(events)
        self.calls = []

    def step(self, **kwargs):
        self.calls.append(kwargs)
        self.last_event = self.events.pop(0)
        return self.last_event


class AI2ThorEnvironmentTests(unittest.TestCase):
    def _environment(self, controller, **kwargs):
        return AI2ThorEnvironment(
            controller=controller,
            target_object_id=TARGET_ID,
            initial_target_location="source",
            destination_location="destination",
            locations={
                "source": SOURCE_ID,
                "moved": MOVED_ID,
                "destination": DESTINATION_ID,
            },
            **kwargs,
        )

    def test_implements_agent_environment_protocol_and_maps_navigation(self) -> None:
        controller = ScriptedController(
            thor_event(objects=[target(visible=False)]),
            thor_event(action_return=[POSE]),
            thor_event(objects=[target(visible=True)]),
        )
        environment = self._environment(controller)
        self.assertIsInstance(environment, AgentEnvironment)

        result = environment.step(Action(ActionKind.NAVIGATE, "source"))
        self.assertTrue(result.success)
        self.assertEqual(environment.agent_location, "source")
        self.assertEqual(controller.calls[0]["action"], "GetInteractablePoses")
        self.assertEqual(controller.calls[0]["objectId"], SOURCE_ID)
        self.assertEqual(controller.calls[1]["action"], "TeleportFull")
        self.assertIn(TARGET_ID, result.observation.visible_objects)

    def test_failed_pickup_at_inspected_location_contradicts_shared_memory(self) -> None:
        initial = thor_event(objects=[target(visible=True)])
        controller = ScriptedController(
            initial,
            thor_event(action_return=[POSE]),
            thor_event(objects=[target(visible=True)]),
            thor_event(success=False, objects=[target(visible=False)], error="Object not visible"),
        )
        environment = self._environment(controller)
        memory = BeliefMemory(target_object=TARGET_ID)
        environment.step(Action(ActionKind.NAVIGATE, "source"))
        inspected = environment.step(Action(ActionKind.INSPECT, "source"))
        memory.ingest_result(Action(ActionKind.INSPECT, "source"), inspected)
        failed = environment.step(Action(ActionKind.PICK_UP, TARGET_ID))
        memory.ingest_result(Action(ActionKind.PICK_UP, TARGET_ID), failed)

        self.assertFalse(failed.success)
        self.assertIn("Object not visible", failed.message)
        self.assertIsNotNone(memory.record)
        self.assertEqual(memory.record.freshness, Freshness.CONTRADICTED)

    def test_place_uses_destination_receptacle_and_completes_task(self) -> None:
        controller = ScriptedController(
            thor_event(objects=[target(visible=False)]),
            thor_event(success=True, objects=[target(visible=False)]),
        )
        environment = self._environment(controller)
        environment.agent_location = "destination"
        environment.holding = TARGET_ID

        result = environment.step(Action(ActionKind.PLACE, "destination"))
        self.assertTrue(result.success)
        self.assertTrue(environment.task_complete())
        self.assertEqual(controller.calls[0]["action"], "PutObject")
        self.assertEqual(controller.calls[0]["objectId"], DESTINATION_ID)

    def test_oracle_location_uses_target_parent_receptacle_mapping(self) -> None:
        controller = ScriptedController(
            thor_event(objects=[target(visible=False, parents=[SOURCE_ID])])
        )
        self.assertEqual(self._environment(controller).hidden_target_location(), "source")

    def test_fast_slow_agent_recovers_through_dynamic_thor_event(self) -> None:
        controller = ScriptedController(
            thor_event(objects=[target(visible=False)]),
            thor_event(action_return=[POSE]),
            thor_event(objects=[target(visible=True, parents=[SOURCE_ID])]),
            thor_event(action_return=[{"x": 0, "y": 1, "z": 0}]),
            thor_event(objects=[target(visible=False, parents=[MOVED_ID])]),
            thor_event(success=False, objects=[target(visible=False, parents=[MOVED_ID])], error="Object not visible"),
            thor_event(action_return=[POSE]),
            thor_event(objects=[target(visible=True, parents=[MOVED_ID])]),
            thor_event(success=True, objects=[target(visible=False)]),
            thor_event(action_return=[POSE]),
            thor_event(objects=[target(visible=False)]),
            thor_event(success=True, objects=[target(visible=False, parents=[DESTINATION_ID])]),
        )
        environment = self._environment(
            controller,
            dynamic_event=True,
            moved_target_location="moved",
        )

        result = FastSlowAgent().run(environment)

        self.assertTrue(result.success)
        self.assertTrue(environment.event_triggered)
        self.assertEqual(result.slow_calls, 1)
        self.assertTrue(any(entry.mode == "delegate" for entry in result.trace))
        self.assertEqual(
            [call["action"] for call in controller.calls],
            [
                "GetInteractablePoses",
                "TeleportFull",
                "GetSpawnCoordinatesAboveReceptacle",
                "PlaceObjectAtPoint",
                "PickupObject",
                "GetInteractablePoses",
                "TeleportFull",
                "PickupObject",
                "GetInteractablePoses",
                "TeleportFull",
                "PutObject",
            ],
        )


if __name__ == "__main__":
    unittest.main()
