import unittest

from fast_slow_agent.models import Freshness
from fast_slow_agent.thor_memory import ThorMemoryBridge


class FakeEvent:
    def __init__(self, *, visible, success, error_message=""):
        self.metadata = {
            "lastActionSuccess": success,
            "errorMessage": error_message,
            "objects": [{"objectId": "Mug|target", "visible": visible}],
        }


class ThorMemoryBridgeTests(unittest.TestCase):
    def test_failed_pickup_at_observed_location_marks_memory_contradicted(self):
        bridge = ThorMemoryBridge(target_object_id="Mug|target")
        bridge.ingest_observation(FakeEvent(visible=True, success=True), location="counter_a", step=1)
        self.assertEqual(bridge.memory.record.freshness, Freshness.FRESH)

        bridge.ingest_pickup_result(
            FakeEvent(visible=False, success=False, error_message="Target not visible"),
            location="counter_a",
            step=2,
        )
        self.assertEqual(bridge.memory.record.freshness, Freshness.CONTRADICTED)
        self.assertEqual(bridge.memory.record.confidence, 0.18)


if __name__ == "__main__":
    unittest.main()
