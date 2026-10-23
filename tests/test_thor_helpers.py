import unittest

from fast_slow_agent.thor_demo import AI2ThorFastSlowDemo


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


if __name__ == "__main__":
    unittest.main()
