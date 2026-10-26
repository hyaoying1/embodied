"""Optional physical integration checks; opt in with RUN_THOR_INTEGRATION=1."""

import os
import unittest

from fast_slow_agent.thor_agent_demo import AI2ThorAgentDemo
from fast_slow_agent.thor_local_demo import LocalVisionThorDemo


@unittest.skipUnless(
    os.environ.get("RUN_THOR_INTEGRATION") == "1",
    "Set RUN_THOR_INTEGRATION=1 to launch AI2-THOR during tests.",
)
class ThorIntegrationTests(unittest.TestCase):
    def test_shared_agent_loop_recovers_after_real_target_move(self):
        result = AI2ThorAgentDemo().run()
        self.assertTrue(result.success)
        self.assertEqual(result.episode.slow_calls, 1)
        self.assertTrue(any(entry.mode == "delegate" for entry in result.episode.trace))

    def test_local_perception_recovers_after_real_target_move(self):
        result = LocalVisionThorDemo().run()
        self.assertTrue(result.success)
        self.assertEqual(result.selected_counter_id, result.moved_counter_id)
        self.assertGreater(result.scanned_views, 0)


if __name__ == "__main__":
    unittest.main()
