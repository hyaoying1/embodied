"""AI2-THOR 3D integration for the fast-slow delegation demonstration.

``oracle`` mode is deterministic and used for physical smoke tests. ``vlm`` mode
uses first-person candidate views and an explicitly configured OpenAI-compatible
vision endpoint to choose the recovery area. No network request is made unless
``vlm`` mode is selected and the required environment variables are present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .learning import TinyDelegationModel
from .models import BeliefRecord, Freshness
from .vision_slow_brain import (
    OpenAICompatibleVisionClient,
    VisionCandidate,
    VisionSearchSlowBrain,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/thor")
TASK_DESCRIPTION = "Find the blue mug after it has moved, then place it on a different countertop."


@dataclass(frozen=True)
class ThorDemoResult:
    scene: str
    source_counter_id: str
    moved_counter_id: str
    selected_counter_id: str
    destination_counter_id: str
    slow_brain_mode: str
    slow_brain_detail: str
    delegate_probability: float
    source_pickup_failed: bool
    recovered_pickup_succeeded: bool
    placement_succeeded: bool
    frames: tuple[Path, ...]

    @property
    def success(self) -> bool:
        return (
            self.source_pickup_failed
            and self.recovered_pickup_succeeded
            and self.placement_succeeded
        )


class AI2ThorFastSlowDemo:
    """Run one real 3D target-move and recovery episode in FloorPlan1.

    The VLM decides *which counter area to search*. The final movement to an
    AI2-THOR interaction pose remains a simulator-provided low-level controller
    bridge; it is not claimed to be a learned robot grasp policy.
    """

    def __init__(
        self,
        *,
        scene: str = "FloorPlan1",
        width: int = 300,
        height: int = 300,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        slow_brain_mode: str = "oracle",
        vision_slow_brain: VisionSearchSlowBrain | None = None,
        slow_request_budget: int = 3,
        slow_image_budget_per_request: int = 4,
    ) -> None:
        if slow_brain_mode not in {"oracle", "vlm"}:
            raise ValueError("slow_brain_mode must be 'oracle' or 'vlm'.")
        if slow_brain_mode == "vlm" and vision_slow_brain is None:
            raise ValueError("VLM mode requires a VisionSlowBrain instance.")
        if slow_request_budget <= 0 or slow_image_budget_per_request <= 0:
            raise ValueError("Slow Brain request and image budgets must be positive.")
        self.scene = scene
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.slow_brain_mode = slow_brain_mode
        self.vision_slow_brain = vision_slow_brain
        self.slow_request_budget = slow_request_budget
        self.slow_image_budget_per_request = slow_image_budget_per_request
        self.delegation_model = TinyDelegationModel.default()

    @classmethod
    def with_vlm_from_environment(cls, **kwargs: Any) -> "AI2ThorFastSlowDemo":
        client = OpenAICompatibleVisionClient.from_environment()
        request_budget = kwargs.get("slow_request_budget", 3)
        image_budget = kwargs.get("slow_image_budget_per_request", 4)
        return cls(
            slow_brain_mode="vlm",
            vision_slow_brain=VisionSearchSlowBrain(
                client,
                max_requests=request_budget,
                max_images_per_candidate=image_budget,
            ),
            **kwargs,
        )

    def run(self) -> ThorDemoResult:
        controller = self._create_controller()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            event = controller.last_event
            mug, source, moved, destination = self._discover_objects(event.metadata["objects"])
            counters = [source, moved, destination]
            source_pose = self._first_interactable_pose(controller, mug["objectId"])
            self._teleport(controller, source_pose)
            from .thor_memory import ThorMemoryBridge

            memory_bridge = ThorMemoryBridge(target_object_id=mug["objectId"])
            memory_bridge.ingest_observation(
                controller.last_event,
                location=source["objectId"],
                step=1,
            )
            if memory_bridge.memory.record is None:
                raise RuntimeError("Target was not visible at its initial interaction pose.")
            frames = [self._save_frame(controller.last_event.frame, "01_initial.png")]

            # The target moves after a fresh, observation-backed memory is established.
            self._move_mug_to_receptacle(controller, mug["objectId"], moved["objectId"])
            frames.append(self._save_frame(controller.last_event.frame, "02_target_moved.png"))

            # Replaying the pre-move interaction pose makes the physical pickup fail.
            self._teleport(controller, source_pose)
            failed_pickup_event = controller.step(
                action="PickupObject",
                objectId=mug["objectId"],
                raise_for_failure=False,
            )
            failed_pickup_result = memory_bridge.ingest_pickup_result(
                failed_pickup_event,
                location=source["objectId"],
                step=2,
            )
            source_pickup_failed = not failed_pickup_result.success
            frames.append(self._save_frame(failed_pickup_event.frame, "03_stale_pickup_failed.png"))
            if not source_pickup_failed:
                raise RuntimeError("Expected a stale-memory pickup to fail in the source area.")

            contradicted_memory = memory_bridge.memory.record
            if contradicted_memory is None or contradicted_memory.freshness is not Freshness.CONTRADICTED:
                raise RuntimeError("Failed pickup did not invalidate the observation-backed memory.")
            delegate_probability = self.delegation_model.predict_probability(
                contradicted_memory,
                current_step=2,
            )
            if not self.delegation_model.should_delegate(contradicted_memory, current_step=2):
                raise RuntimeError("Fast Brain did not delegate after a contradicted 3D observation.")

            selected, slow_detail, search_frames = self._select_recovery_counter(
                controller,
                counters=counters,
                oracle_moved_counter=moved,
            )
            frames.extend(search_frames)
            if selected["objectId"] != moved["objectId"]:
                raise RuntimeError(
                    "Slow Brain selected the wrong counter area. "
                    f"selected={selected['objectId']}; expected={moved['objectId']}; detail={slow_detail}"
                )

            # Low-level simulator bridge: once the VLM selects an area, use the
            # known target instance ID and AI2-THOR's interaction pose to execute pickup.
            recovered_pose = self._first_interactable_pose(controller, mug["objectId"])
            self._teleport(controller, recovered_pose)
            recovered_pickup_event = controller.step(
                action="PickupObject",
                objectId=mug["objectId"],
                raise_for_failure=False,
            )
            recovered_pickup_succeeded = recovered_pickup_event.metadata["lastActionSuccess"]
            frames.append(self._save_frame(recovered_pickup_event.frame, "04_recovered_pickup.png"))
            if not recovered_pickup_succeeded:
                raise RuntimeError(self._failure_message(recovered_pickup_event, "Could not pick up moved mug"))

            destination_pose = self._first_interactable_pose(controller, destination["objectId"])
            self._teleport(controller, destination_pose)
            placement_event = controller.step(
                action="PutObject",
                objectId=destination["objectId"],
                raise_for_failure=False,
            )
            placement_succeeded = placement_event.metadata["lastActionSuccess"]
            frames.append(self._save_frame(placement_event.frame, "05_placed.png"))
            if not placement_succeeded:
                raise RuntimeError(self._failure_message(placement_event, "Could not place mug"))

            return ThorDemoResult(
                scene=self.scene,
                source_counter_id=source["objectId"],
                moved_counter_id=moved["objectId"],
                selected_counter_id=selected["objectId"],
                destination_counter_id=destination["objectId"],
                slow_brain_mode=self.slow_brain_mode,
                slow_brain_detail=slow_detail,
                delegate_probability=delegate_probability,
                source_pickup_failed=source_pickup_failed,
                recovered_pickup_succeeded=recovered_pickup_succeeded,
                placement_succeeded=placement_succeeded,
                frames=tuple(frames),
            )
        finally:
            controller.stop()

    def _select_recovery_counter(
        self,
        controller: Any,
        *,
        counters: list[dict[str, Any]],
        oracle_moved_counter: dict[str, Any],
    ) -> tuple[dict[str, Any], str, list[Path]]:
        if self.slow_brain_mode == "oracle":
            return oracle_moved_counter, "oracle selected the known moved-target counter", []

        if self.vision_slow_brain is None:
            raise RuntimeError("VLM mode was selected without a VisionSlowBrain.")
        candidates, frames = self._capture_candidate_views(controller, counters)
        result = self.vision_slow_brain.locate(task=TASK_DESCRIPTION, candidates=candidates)
        if result.selected_label is None:
            error_summary = "; ".join(
                f"{label}: {error}" for label, error in result.candidate_errors.items()
            ) or "none"
            raise RuntimeError(
                "VLM recovery search abstained. "
                f"reason={result.abstain_reason}; candidate_errors={error_summary}; "
                f"assessed_candidates={len(result.assessments)}"
            )
        selected = next(counter for counter in counters if counter["objectId"] == result.selected_label)
        detail = (
            f"VLM selected {result.selected_label} with confidence={result.confidence:.2f}; "
            f"rationale={result.rationale}; candidate_errors={len(result.candidate_errors)}"
        )
        return selected, detail, frames

    def _capture_candidate_views(
        self,
        controller: Any,
        counters: list[dict[str, Any]],
    ) -> tuple[list[VisionCandidate], list[Path]]:
        candidates: list[VisionCandidate] = []
        frames: list[Path] = []
        for index, counter in enumerate(counters, start=1):
            poses = self._sample_search_poses(
                self._interactable_poses(controller, counter["objectId"]),
                max_views=8,
            )
            views: list[Path] = []
            # Eight evenly spaced poses give the VLM diverse views of each area.
            # Coverage diagnostics show the moved-target area exposes the mug in
            # most of these views while non-target areas expose none.
            for view_index, pose in enumerate(poses, start=1):
                self._teleport(controller, pose)
                path = self._save_frame(
                    controller.last_event.frame,
                    f"search_counter_{index}_view_{view_index}.png",
                )
                views.append(path)
                frames.append(path)
            candidates.append(VisionCandidate(label=counter["objectId"], image_paths=tuple(views)))
        return candidates, frames

    @staticmethod
    def _sample_search_poses(poses: list[dict[str, Any]], *, max_views: int) -> list[dict[str, Any]]:
        if max_views <= 0:
            raise ValueError("max_views must be positive.")
        if len(poses) <= max_views:
            return poses
        indices = {round(index * (len(poses) - 1) / (max_views - 1)) for index in range(max_views)}
        return [pose for index, pose in enumerate(poses) if index in indices]

    def _create_controller(self) -> Any:
        try:
            from ai2thor.controller import Controller
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "AI2-THOR is not installed. Run `.venv/bin/python -m pip install ai2thor==5.0.0`."
            ) from error
        return Controller(
            scene=self.scene,
            width=self.width,
            height=self.height,
            quality="Very Low",
        )

    @staticmethod
    def _discover_objects(objects: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        object_list = list(objects)
        mug = next(
            (obj for obj in object_list if obj["objectType"] == "Mug" and obj.get("pickupable")),
            None,
        )
        countertops = [
            obj for obj in object_list if obj["objectType"] == "CounterTop" and obj.get("receptacle")
        ]
        if mug is None or len(countertops) < 3:
            raise RuntimeError("FloorPlan1 must contain one pickupable mug and three countertops.")

        source = min(countertops, key=lambda counter: _squared_distance(mug["position"], counter["position"]))
        moved = max(countertops, key=lambda counter: _squared_distance(mug["position"], counter["position"]))
        destination = next(
            counter
            for counter in countertops
            if counter["objectId"] not in {source["objectId"], moved["objectId"]}
        )
        return mug, source, moved, destination

    @staticmethod
    def _interactable_poses(controller: Any, object_id: str) -> list[dict[str, Any]]:
        event = controller.step(
            action="GetInteractablePoses",
            objectId=object_id,
            rotations=[0, 45, 90, 135, 180, 225, 270, 315],
            horizons=[0],
            standings=[True],
        )
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(AI2ThorFastSlowDemo._failure_message(event, "Could not query interactable poses"))
        poses = event.metadata.get("actionReturn")
        if not poses:
            raise RuntimeError(f"No interactable pose found for {object_id}.")
        return poses

    @classmethod
    def _first_interactable_pose(cls, controller: Any, object_id: str) -> dict[str, Any]:
        return cls._interactable_poses(controller, object_id)[0]

    @staticmethod
    def _teleport(controller: Any, pose: dict[str, Any]) -> None:
        event = controller.step(
            action="TeleportFull",
            position={"x": pose["x"], "y": pose["y"], "z": pose["z"]},
            rotation={"x": 0, "y": pose["rotation"], "z": 0},
            horizon=pose["horizon"],
            standing=pose["standing"],
        )
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(AI2ThorFastSlowDemo._failure_message(event, "Could not teleport"))

    @staticmethod
    def _move_mug_to_receptacle(controller: Any, mug_id: str, receptacle_id: str) -> None:
        spawn_event = controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_id,
            anywhere=True,
        )
        if not spawn_event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(
                AI2ThorFastSlowDemo._failure_message(spawn_event, "Could not query receptacle spawn points")
            )
        points = spawn_event.metadata.get("actionReturn")
        if not points:
            raise RuntimeError(f"No clear spawn point found for {receptacle_id}.")
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId=mug_id,
            positions=points,
            forceKinematic=True,
            raise_for_failure=False,
        )
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(AI2ThorFastSlowDemo._failure_message(event, "Could not move mug"))

    def _save_frame(self, frame: Any, filename: str) -> Path:
        try:
            from PIL import Image
        except ModuleNotFoundError as error:
            raise RuntimeError("Pillow is required by the installed AI2-THOR backend to save frames.") from error
        path = self.output_dir / filename
        Image.fromarray(frame).save(path)
        return path

    @staticmethod
    def _failure_message(event: Any, prefix: str) -> str:
        return f"{prefix}: {event.metadata.get('errorMessage', 'unknown AI2-THOR error')}"


def _squared_distance(first: dict[str, float], second: dict[str, float]) -> float:
    return (first["x"] - second["x"]) ** 2 + (first["z"] - second["z"]) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fast-Slow AI2-THOR 3D episode.")
    parser.add_argument(
        "--slow-brain",
        choices=("oracle", "vlm"),
        default="oracle",
        help="Use deterministic oracle recovery or a configured VLM visual search.",
    )
    args = parser.parse_args()
    demo = (
        AI2ThorFastSlowDemo.with_vlm_from_environment()
        if args.slow_brain == "vlm"
        else AI2ThorFastSlowDemo()
    )
    result = demo.run()
    print("AI2-THOR Fast-Slow 3D episode")
    print(f"scene={result.scene}")
    print(f"slow_brain_mode={result.slow_brain_mode}")
    print(f"source={result.source_counter_id}")
    print(f"moved_to={result.moved_counter_id}")
    print(f"selected={result.selected_counter_id}")
    print(f"destination={result.destination_counter_id}")
    print(f"delegate_probability={result.delegate_probability:.2f}")
    print(f"slow_brain_detail={result.slow_brain_detail}")
    print(f"success={result.success}")
    print("frames:")
    for frame in result.frames:
        print(f"  {frame.resolve()}")


if __name__ == "__main__":
    main()
