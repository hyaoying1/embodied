"""AI2-THOR candidate-frame provider for VisionSearchSlowPlanner."""

from __future__ import annotations

from pathlib import Path

from .ai2thor_environment import AI2ThorEnvironment
from .interfaces import AgentEnvironment, SlowPlannerError
from .vision_slow_brain import VisionCandidate


class AI2ThorVisionCandidateProvider:
    """Capture a bounded set of first-person frames for each mapped THOR area."""

    def __init__(self, *, output_dir: Path, max_views_per_candidate: int = 4) -> None:
        if max_views_per_candidate <= 0:
            raise ValueError("Maximum views per candidate must be positive.")
        self.output_dir = output_dir
        self.max_views_per_candidate = max_views_per_candidate

    def __call__(self, environment: AgentEnvironment) -> list[VisionCandidate]:
        if not isinstance(environment, AI2ThorEnvironment):
            raise SlowPlannerError("AI2-THOR vision provider requires AI2ThorEnvironment.")
        try:
            from PIL import Image
        except ModuleNotFoundError as error:
            raise SlowPlannerError("Pillow is required to capture THOR vision candidates.") from error
        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[VisionCandidate] = []
        for index, (label, object_id) in enumerate(environment.locations.items(), start=1):
            event = environment.controller.step(
                action="GetInteractablePoses",
                objectId=object_id,
                rotations=[0, 90, 180, 270],
                horizons=[0],
                standings=[True],
            )
            if not event.metadata.get("lastActionSuccess", False):
                raise SlowPlannerError(f"Could not capture poses for {label}: {event.metadata.get('errorMessage', '')}")
            poses = (event.metadata.get("actionReturn") or [])[: self.max_views_per_candidate]
            if not poses:
                raise SlowPlannerError(f"No candidate poses available for {label}.")
            frames: list[Path] = []
            for view_index, pose in enumerate(poses, start=1):
                event = environment.controller.step(
                    action="TeleportFull",
                    position={"x": pose["x"], "y": pose["y"], "z": pose["z"]},
                    rotation={"x": 0, "y": pose["rotation"], "z": 0},
                    horizon=pose["horizon"],
                    standing=pose["standing"],
                )
                if not event.metadata.get("lastActionSuccess", False):
                    raise SlowPlannerError(f"Could not capture view for {label}: {event.metadata.get('errorMessage', '')}")
                path = self.output_dir / f"candidate_{index}_view_{view_index}.png"
                Image.fromarray(event.frame).save(path)
                frames.append(path)
            candidates.append(VisionCandidate(label=label, image_paths=tuple(frames)))
        return candidates
