import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fast_slow_agent.vision_slow_brain import (
    OpenAICompatibleVisionClient,
    VisionAssessment,
    VisionCandidate,
    VisionSearchSlowBrain,
    parse_vision_assessment,
)


class FakeVisionClient:
    def __init__(self, assessments):
        self.assessments = assessments

    def assess_candidate(self, *, task, candidate):
        return self.assessments[candidate.label]


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class VisionSlowBrainTests(unittest.TestCase):
    def test_parses_json_assessment(self) -> None:
        assessment = parse_vision_assessment(
            '```json\n{"target_visible": true, "confidence": 0.87, "rationale": "Blue mug visible."}\n```'
        )
        self.assertTrue(assessment.target_visible)
        self.assertEqual(assessment.confidence, 0.87)
        self.assertEqual(assessment.rationale, "Blue mug visible.")

    def test_rejects_boolean_confidence_in_json_assessment(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "confidence"):
            parse_vision_assessment(
                '{"target_visible": true, "confidence": true, "rationale": "Blue mug visible."}'
            )

    def test_selects_highest_confidence_visible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"placeholder")
            candidates = [
                VisionCandidate("counter_a", (image,)),
                VisionCandidate("counter_b", (image,)),
                VisionCandidate("counter_c", (image,)),
            ]
            client = FakeVisionClient(
                {
                    "counter_a": VisionAssessment(False, 0.91, "No mug."),
                    "counter_b": VisionAssessment(True, 0.72, "Possible mug."),
                    "counter_c": VisionAssessment(True, 0.88, "Blue mug visible."),
                }
            )
            result = VisionSearchSlowBrain(client).locate(
                task="Find the blue mug.",
                candidates=candidates,
            )
        self.assertEqual(result.selected_label, "counter_c")
        self.assertEqual(result.confidence, 0.88)

    def test_rejects_duplicate_candidate_labels_before_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"placeholder")
            with self.assertRaisesRegex(ValueError, "labels must be unique"):
                VisionSearchSlowBrain(FakeVisionClient({})).locate(
                    task="Find the blue mug.",
                    candidates=[
                        VisionCandidate("counter_a", (image,)),
                        VisionCandidate("counter_a", (image,)),
                    ],
                )

    def test_rejects_non_finite_client_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"placeholder")
            client = FakeVisionClient(
                {"counter_a": VisionAssessment(True, float("nan"), "Blue mug visible.")}
            )
            with self.assertRaisesRegex(ValueError, "counter_a.*invalid confidence"):
                VisionSearchSlowBrain(client).locate(
                    task="Find the blue mug.",
                    candidates=[VisionCandidate("counter_a", (image,))],
                )

    def test_fails_when_no_candidate_is_visually_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"placeholder")
            client = FakeVisionClient(
                {"counter_a": VisionAssessment(False, 0.9, "No mug.")}
            )
            with self.assertRaisesRegex(RuntimeError, "did not identify"):
                VisionSearchSlowBrain(client).locate(
                    task="Find the blue mug.",
                    candidates=[VisionCandidate("counter_a", (image,))],
                )


class OpenAICompatibleVisionClientTests(unittest.TestCase):
    def test_builds_prompt_from_task_without_hardcoded_target(self) -> None:
        response = FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"target_visible": true, "confidence": 0.9, '
                                '"rationale": "Green lettuce visible."}'
                            )
                        }
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"placeholder")
            client = OpenAICompatibleVisionClient(api_key="test-key", model="test-model")
            with patch(
                "fast_slow_agent.vision_slow_brain.urllib.request.urlopen",
                return_value=response,
            ) as mock_urlopen:
                assessment = client.assess_candidate(
                    task="Find the green lettuce and place it on the table.",
                    candidate=VisionCandidate("counter_a", (image,)),
                )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        prompt = payload["messages"][1]["content"][0]["text"]
        self.assertTrue(assessment.target_visible)
        self.assertEqual(assessment.confidence, 0.9)
        self.assertEqual(payload["model"], "test-model")
        self.assertIn("Find the green lettuce and place it on the table.", prompt)
        self.assertIn("Candidate area label: counter_a", prompt)
        self.assertIn("target described by the task", prompt)
        self.assertNotIn("BLUE MUG", prompt)
        self.assertEqual(len(payload["messages"][1]["content"]), 2)


class VisionBudgetTests(unittest.TestCase):
    def test_caps_request_and_image_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_paths = []
            for index in range(5):
                image = Path(directory) / f"frame-{index}.png"
                image.write_bytes(b"placeholder")
                image_paths.append(image)
            client = FakeVisionClient(
                {
                    "counter_a": VisionAssessment(False, 0.9, "No mug."),
                    "counter_b": VisionAssessment(True, 0.8, "Mug visible."),
                    "counter_c": VisionAssessment(True, 0.7, "Mug visible."),
                }
            )
            result = VisionSearchSlowBrain(
                client,
                max_requests=2,
                max_images_per_candidate=3,
            ).locate(
                task="Find the mug.",
                candidates=[
                    VisionCandidate("counter_a", tuple(image_paths)),
                    VisionCandidate("counter_b", tuple(image_paths)),
                    VisionCandidate("counter_c", tuple(image_paths)),
                ],
            )
        self.assertEqual(result.selected_label, "counter_b")
        self.assertEqual(result.requests_used, 2)
        self.assertEqual(result.images_used, 6)


class VisionResponseValidationTests(unittest.TestCase):
    def test_rejects_json_array_containing_an_assessment(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be an object"):
            parse_vision_assessment(
                '[{"target_visible": true, "confidence": 0.9, "rationale": "Visible."}]'
            )

    def test_rejects_non_text_model_content(self) -> None:
        response = FakeHTTPResponse(
            {"choices": [{"message": {"content": None}}]}
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"placeholder")
            client = OpenAICompatibleVisionClient(api_key="test-key", model="test-model")
            with patch(
                "fast_slow_agent.vision_slow_brain.urllib.request.urlopen",
                return_value=response,
            ):
                with self.assertRaisesRegex(RuntimeError, "content must be a string"):
                    client.assess_candidate(
                        task="Find the green lettuce.",
                        candidate=VisionCandidate("counter_a", (image,)),
                    )


if __name__ == "__main__":
    unittest.main()
