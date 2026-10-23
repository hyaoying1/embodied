"""Fully local AI2-THOR perception episode with no external visual model.

The target is a green Lettuce in FloorPlan1. Once the Fast Brain detects that
its memory is contradicted, the Slow Brain performs an active local search: it
visits candidate countertop viewpoints and uses AI2-THOR's local ``visible``
sensor metadata to identify the target area. This is simulator perception, not a
learned visual model, and is labeled as such in output and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .learning import TinyDelegationModel
from .models import BeliefRecord, Freshness
from .thor_demo import AI2ThorFastSlowDemo, DEFAULT_OUTPUT_DIR


LOCAL_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "local-perception"
TARGET_TYPE = "Lettuce"


@dataclass(frozen=True)
class LocalVisionThorResult:
    selected_counter_id: str
    moved_counter_id: str
    delegate_probability: float
    local_confidence: float
    source_pickup_failed: bool
    recovered_pickup_succeeded: bool
    placement_succeeded: bool
    scanned_views: int
    frames: tuple[Path, ...]

    @property
    def success(self) -> bool:
        return (
            self.selected_counter_id == self.moved_counter_id
            and self.source_pickup_failed
            and self.recovered_pickup_succeeded
            and self.placement_succeeded
        )


class LocalVisionThorDemo:
    """Use AI2-THOR's local visibility sensor as an offline Slow Brain."""

    def __init__(self, *, output_dir: Path = LOCAL_OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        self.bridge = AI2ThorFastSlowDemo(output_dir=output_dir)
        self.fast_brain = TinyDelegationModel.default()

    def run(self) -> LocalVisionThorResult:
        controller = self.bridge._create_controller()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            target, source, moved, destination = self._discover_target_and_counters(
                controller.last_event.metadata["objects"]
            )
            source_pose = self.bridge._first_interactable_pose(controller, target["objectId"])
            self.bridge._teleport(controller, source_pose)
            from .thor_memory import ThorMemoryBridge

            memory_bridge = ThorMemoryBridge(target_object_id=target["objectId"])
            memory_bridge.ingest_observation(
                controller.last_event,
                location=source["objectId"],
                step=1,
            )
            if memory_bridge.memory.record is None:
                raise RuntimeError("Lettuce was not visible at its initial interaction pose.")
            frames = [self.bridge._save_frame(controller.last_event.frame, "01_initial_lettuce.png")]

            self.bridge._move_mug_to_receptacle(controller, target["objectId"], moved["objectId"])
            frames.append(self.bridge._save_frame(controller.last_event.frame, "02_lettuce_moved.png"))

            self.bridge._teleport(controller, source_pose)
            failed_pickup = controller.step(
                action="PickupObject",
                objectId=target["objectId"],
                raise_for_failure=False,
            )
            failed_pickup_result = memory_bridge.ingest_pickup_result(
                failed_pickup,
                location=source["objectId"],
                step=2,
            )
            source_pickup_failed = not failed_pickup_result.success
            frames.append(self.bridge._save_frame(failed_pickup.frame, "03_stale_pickup_failed.png"))
            if not source_pickup_failed:
                raise RuntimeError("Expected old-location lettuce pickup to fail.")

            memory = memory_bridge.memory.record
            if memory is None or memory.freshness is not Freshness.CONTRADICTED:
                raise RuntimeError("Failed pickup did not invalidate the observation-backed lettuce memory.")
            delegate_probability = self.fast_brain.predict_probability(memory, current_step=2)
            if not self.fast_brain.should_delegate(memory, current_step=2):
                raise RuntimeError("Fast Brain did not delegate the contradicted lettuce memory.")

            selected, scanned_views, search_frames = self._scan_candidate_areas(
                controller,
                target_id=target["objectId"],
                counters=[source, moved, destination],
            )
            frames.extend(search_frames)
            if selected["objectId"] != moved["objectId"]:
                raise RuntimeError(
                    "Local simulator perception selected the wrong area. "
                    f"selected={selected['objectId']}; expected={moved['objectId']}"
                )

            # Simulator low-level interaction bridge after local area selection.
            target_pose = self.bridge._first_interactable_pose(controller, target["objectId"])
            self.bridge._teleport(controller, target_pose)
            recovered_pickup = controller.step(
                action="PickupObject",
                objectId=target["objectId"],
                raise_for_failure=False,
            )
            recovered_pickup_succeeded = recovered_pickup.metadata["lastActionSuccess"]
            frames.append(self.bridge._save_frame(recovered_pickup.frame, "04_recovered_pickup.png"))
            if not recovered_pickup_succeeded:
                raise RuntimeError(self.bridge._failure_message(recovered_pickup, "Could not pick up lettuce"))

            destination_pose = self.bridge._first_interactable_pose(controller, destination["objectId"])
            self.bridge._teleport(controller, destination_pose)
            placed = controller.step(
                action="PutObject",
                objectId=destination["objectId"],
                raise_for_failure=False,
            )
            placement_succeeded = placed.metadata["lastActionSuccess"]
            frames.append(self.bridge._save_frame(placed.frame, "05_placed.png"))
            if not placement_succeeded:
                raise RuntimeError(self.bridge._failure_message(placed, "Could not place lettuce"))

            return LocalVisionThorResult(
                selected_counter_id=selected["objectId"],
                moved_counter_id=moved["objectId"],
                delegate_probability=delegate_probability,
                local_confidence=1.0,
                source_pickup_failed=source_pickup_failed,
                recovered_pickup_succeeded=recovered_pickup_succeeded,
                placement_succeeded=placement_succeeded,
                scanned_views=scanned_views,
                frames=tuple(frames),
            )
        finally:
            controller.stop()

    def _scan_candidate_areas(
        self,
        controller: Any,
        *,
        target_id: str,
        counters: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], int, list[Path]]:
        """Actively inspect candidate views using local simulator perception only."""
        frames: list[Path] = []
        scanned_views = 0
        for counter_index, counter in enumerate(counters, start=1):
            poses = self.bridge._sample_search_poses(
                self.bridge._interactable_poses(controller, counter["objectId"]),
                max_views=8,
            )
            for view_index, pose in enumerate(poses, start=1):
                self.bridge._teleport(controller, pose)
                event = controller.last_event
                target = next(obj for obj in event.metadata["objects"] if obj["objectId"] == target_id)
                is_visible = bool(target.get("visible"))
                path = self.bridge._save_frame(
                    event.frame,
                    f"search_counter_{counter_index}_view_{view_index}_visible_{int(is_visible)}.png",
                )
                frames.append(path)
                scanned_views += 1
                if is_visible:
                    return counter, scanned_views, frames
        raise RuntimeError("Local simulator perception could not find the target in candidate areas.")

    @staticmethod
    def _discover_target_and_counters(
        objects: Iterable[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        object_list = list(objects)
        target = next(
            (obj for obj in object_list if obj["objectType"] == TARGET_TYPE and obj.get("pickupable")),
            None,
        )
        countertops = [
            obj for obj in object_list if obj["objectType"] == "CounterTop" and obj.get("receptacle")
        ]
        if target is None or len(countertops) < 3:
            raise RuntimeError("FloorPlan1 requires a pickupable Lettuce and three countertops.")
        source = min(
            countertops,
            key=lambda counter: _squared_distance(target["position"], counter["position"]),
        )
        moved = max(
            countertops,
            key=lambda counter: _squared_distance(target["position"], counter["position"]),
        )
        destination = next(
            counter
            for counter in countertops
            if counter["objectId"] not in {source["objectId"], moved["objectId"]}
        )
        return target, source, moved, destination


def _squared_distance(first: dict[str, float], second: dict[str, float]) -> float:
    return (first["x"] - second["x"]) ** 2 + (first["z"] - second["z"]) ** 2


def main() -> None:
    result = LocalVisionThorDemo().run()
    print("AI2-THOR local-perception Fast-Slow episode")
    print("slow_brain=ai2thor_local_visibility_sensor")
    print(f"moved_counter={result.moved_counter_id}")
    print(f"selected_counter={result.selected_counter_id}")
    print(f"delegate_probability={result.delegate_probability:.2f}")
    print(f"local_perception_confidence={result.local_confidence:.2f}")
    print(f"scanned_views={result.scanned_views}")
    print(f"success={result.success}")
    print("frames:")
    for frame in result.frames:
        print(f"  {frame.resolve()}")


if __name__ == "__main__":
    main()
