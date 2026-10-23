import unittest

from fast_slow_agent.learning import TinyDelegationModel
from fast_slow_agent.models import BeliefRecord, Freshness


class TinyDelegationModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyDelegationModel.default()

    def test_contradicted_memory_is_a_high_probability_delegation(self) -> None:
        record = BeliefRecord(
            object_name="blue_mug",
            last_known_location="left_cabinet",
            observed_step=2,
            confidence=0.18,
            freshness=Freshness.CONTRADICTED,
        )
        probability = self.model.predict_probability(record, current_step=3)
        self.assertGreater(probability, self.model.threshold)
        self.assertTrue(self.model.should_delegate(record, current_step=3))

    def test_fresh_high_confidence_memory_stays_on_fast_path(self) -> None:
        record = BeliefRecord(
            object_name="blue_mug",
            last_known_location="left_cabinet",
            observed_step=2,
            confidence=0.92,
            freshness=Freshness.FRESH,
        )
        probability = self.model.predict_probability(record, current_step=3)
        self.assertLess(probability, self.model.threshold)
        self.assertFalse(self.model.should_delegate(record, current_step=3))


if __name__ == "__main__":
    unittest.main()
