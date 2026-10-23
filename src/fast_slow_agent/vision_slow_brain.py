"""Configurable VLM-backed Slow Brain for visual candidate search.

No external call occurs unless the caller explicitly constructs an
``OpenAICompatibleVisionClient`` with a key and invokes ``locate``. The client
sends only the selected first-person PNG frames and the fixed task prompt.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class VisionCandidate:
    label: str
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class VisionAssessment:
    target_visible: bool
    confidence: float
    rationale: str


@dataclass(frozen=True)
class VisionSearchResult:
    selected_label: str
    confidence: float
    rationale: str
    assessments: dict[str, VisionAssessment]
    requests_used: int
    images_used: int


class VisionClient(Protocol):
    def assess_candidate(self, *, task: str, candidate: VisionCandidate) -> VisionAssessment:
        """Judge whether the task target is visible in candidate first-person frames."""


def _validate_assessment(assessment: VisionAssessment, *, candidate_label: str) -> None:
    """Enforce the public VisionClient response contract before selection."""
    if not isinstance(assessment, VisionAssessment):
        raise ValueError(
            f"Vision client returned an invalid assessment for candidate {candidate_label!r}."
        )
    if not isinstance(assessment.target_visible, bool):
        raise ValueError(
            f"Vision assessment for candidate {candidate_label!r} must use a boolean target_visible."
        )
    confidence = assessment.confidence
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError(
            f"Vision assessment for candidate {candidate_label!r} has an invalid confidence."
        )
    if not isinstance(assessment.rationale, str):
        raise ValueError(
            f"Vision assessment for candidate {candidate_label!r} must use a string rationale."
        )


class VisionSearchSlowBrain:
    """Bounded visual candidate search with explicit request and image budgets."""

    def __init__(
        self,
        client: VisionClient,
        *,
        max_requests: int = 3,
        max_images_per_candidate: int = 4,
    ) -> None:
        if max_requests <= 0 or max_images_per_candidate <= 0:
            raise ValueError("Visual search budgets must be positive.")
        self.client = client
        self.max_requests = max_requests
        self.max_images_per_candidate = max_images_per_candidate

    def locate(self, *, task: str, candidates: Sequence[VisionCandidate]) -> VisionSearchResult:
        if not candidates:
            raise ValueError("At least one visual candidate is required.")
        bounded_candidates = [
            VisionCandidate(
                label=candidate.label,
                image_paths=candidate.image_paths[: self.max_images_per_candidate],
            )
            for candidate in candidates[: self.max_requests]
        ]
        if any(not candidate.image_paths for candidate in bounded_candidates):
            raise ValueError("Every selected visual candidate must include at least one image.")
        labels = [candidate.label for candidate in bounded_candidates]
        if len(set(labels)) != len(labels):
            raise ValueError("Selected visual candidate labels must be unique.")

        assessments: dict[str, VisionAssessment] = {}
        for candidate in bounded_candidates:
            assessment = self.client.assess_candidate(task=task, candidate=candidate)
            _validate_assessment(assessment, candidate_label=candidate.label)
            assessments[candidate.label] = assessment

        visible = [
            (label, assessment)
            for label, assessment in assessments.items()
            if assessment.target_visible
        ]
        if not visible:
            raise RuntimeError("VLM did not identify the target in any candidate area.")
        label, assessment = max(visible, key=lambda item: item[1].confidence)
        return VisionSearchResult(
            selected_label=label,
            confidence=assessment.confidence,
            rationale=assessment.rationale,
            assessments=assessments,
            requests_used=len(bounded_candidates),
            images_used=sum(len(candidate.image_paths) for candidate in bounded_candidates),
        )


class OpenAICompatibleVisionClient:
    """Minimal OpenAI-compatible multimodal client using only the stdlib.

    Required environment variables for ``from_environment``:

    - ``FAST_SLOW_VLM_API_KEY``
    - ``FAST_SLOW_VLM_MODEL``

    Optional:

    - ``FAST_SLOW_VLM_ENDPOINT`` (defaults to OpenAI chat completions)
    """

    default_endpoint = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = default_endpoint,
        timeout_seconds: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError("A VLM API key is required.")
        if not model:
            raise ValueError("A VLM model name is required.")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleVisionClient":
        api_key = os.environ.get("FAST_SLOW_VLM_API_KEY", "")
        model = os.environ.get("FAST_SLOW_VLM_MODEL", "")
        endpoint = os.environ.get("FAST_SLOW_VLM_ENDPOINT", cls.default_endpoint)
        if not api_key or not model:
            raise RuntimeError(
                "VLM mode requires FAST_SLOW_VLM_API_KEY and FAST_SLOW_VLM_MODEL. "
                "No image request was sent."
            )
        return cls(api_key=api_key, model=model, endpoint=endpoint)

    def assess_candidate(self, *, task: str, candidate: VisionCandidate) -> VisionAssessment:
        if not candidate.image_paths:
            raise ValueError(f"Candidate {candidate.label} has no images.")
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "You are the Slow Brain of an embodied agent. "
                    f"Task: {task}\n"
                    f"Candidate area label: {candidate.label}\n"
                    "Review the first-person images for this candidate. Is the target described "
                    "by the task visibly present? Return only a JSON object with keys target_visible "
                    "(boolean), confidence (number 0 through 1), and rationale (short string). Do "
                    "not assume the target is present when it is not visible."
                ),
            }
        ]
        for image_path in candidate.image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._data_url(image_path),
                        "detail": "low",
                    },
                }
            )

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Respond with a strict JSON object and no Markdown fences.",
                },
                {"role": "user", "content": content},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM request failed with HTTP {error.code}: {response_text}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"VLM network request failed: {error.reason}") from error

        try:
            content_text = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("VLM response did not contain choices[0].message.content.") from error
        if not isinstance(content_text, str):
            raise RuntimeError("VLM response choices[0].message.content must be a string.")
        return parse_vision_assessment(content_text)

    @staticmethod
    def _data_url(image_path: Path) -> str:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


def parse_vision_assessment(content: str) -> VisionAssessment:
    """Parse a strict or fenced JSON object result from a vision provider."""
    normalized = content.strip()
    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        normalized,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if normalized.startswith("```"):
        if not fenced_match:
            raise RuntimeError(f"VLM response was not a valid fenced JSON object: {content!r}")
        serialized_payload = fenced_match.group(1)
    else:
        serialized_payload = normalized

    try:
        payload = json.loads(serialized_payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"VLM returned invalid JSON: {content!r}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("VLM JSON response must be an object.")

    visible = payload.get("target_visible")
    confidence = payload.get("confidence")
    rationale = payload.get("rationale")
    if not isinstance(visible, bool):
        raise RuntimeError("VLM JSON field target_visible must be boolean.")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (float, int))
        or not math.isfinite(confidence)
        or not 0 <= float(confidence) <= 1
    ):
        raise RuntimeError("VLM JSON field confidence must be a finite number between 0 and 1.")
    if not isinstance(rationale, str):
        raise RuntimeError("VLM JSON field rationale must be a string.")
    return VisionAssessment(
        target_visible=visible,
        confidence=float(confidence),
        rationale=rationale,
    )
