"""Fully local, no-model visual client for controlled 3D color-target demos.

This client is intentionally simple: it detects a strongly green target in RGB
frames with Pillow. It is not a general object detector and is kept separate
from the external VLM client so experiment claims remain honest.
"""

from __future__ import annotations

from pathlib import Path

from .vision_slow_brain import VisionAssessment, VisionCandidate


class LocalGreenObjectVisionClient:
    """Detect a green target using local RGB thresholding on candidate PNGs."""

    def __init__(
        self,
        *,
        minimum_green_pixels: int = 80,
        green_minimum: int = 70,
        dominance_ratio: float = 1.25,
    ) -> None:
        self.minimum_green_pixels = minimum_green_pixels
        self.green_minimum = green_minimum
        self.dominance_ratio = dominance_ratio

    def assess_candidate(self, *, task: str, candidate: VisionCandidate) -> VisionAssessment:
        del task  # The controlled local demo knows it is looking for a green lettuce target.
        counts = [self._count_green_pixels(path) for path in candidate.image_paths]
        best_count = max(counts, default=0)
        visible = best_count >= self.minimum_green_pixels
        confidence = min(1.0, best_count / max(self.minimum_green_pixels * 4, 1))
        rationale = (
            f"max_green_pixels={best_count} across {len(counts)} first-person views; "
            f"threshold={self.minimum_green_pixels}"
        )
        return VisionAssessment(
            target_visible=visible,
            confidence=confidence,
            rationale=rationale,
        )

    def _count_green_pixels(self, image_path: Path) -> int:
        try:
            from PIL import Image
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Pillow is required for the local visual detector. Install AI2-THOR dependencies in .venv."
            ) from error
        with Image.open(image_path) as image:
            pixels = image.convert("RGB").getdata()
        return sum(
            1
            for red, green, blue in pixels
            if (
                green >= self.green_minimum
                and green >= red * self.dominance_ratio
                and green >= blue * self.dominance_ratio
            )
        )
