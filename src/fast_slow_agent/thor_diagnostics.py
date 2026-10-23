"""Audit visual candidate coverage in the AI2-THOR target-move scenario.

This is an offline diagnostic only. It uses AI2-THOR metadata to measure whether
the mug is actually visible from sampled candidate viewpoints, allowing us to
validate the image collection strategy before asking a VLM to reason over it.
The metadata signal is never passed to the runtime VLM searcher.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .thor_demo import AI2ThorFastSlowDemo, DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class CandidateCoverage:
    label: str
    object_id: str
    sampled_views: int
    mug_visible_views: int
    image_paths: tuple[str, ...]


@dataclass(frozen=True)
class CoverageReport:
    scene: str
    moved_counter_id: str
    candidates: tuple[CandidateCoverage, ...]
    report_path: str


class ThorVisualCoverageDiagnostics:
    def __init__(self, *, output_dir: Path = DEFAULT_OUTPUT_DIR / "coverage", views_per_candidate: int = 8) -> None:
        self.output_dir = output_dir
        self.views_per_candidate = views_per_candidate
        self.demo = AI2ThorFastSlowDemo(output_dir=output_dir)

    def run(self) -> CoverageReport:
        controller = self.demo._create_controller()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            mug, source, moved, destination = self.demo._discover_objects(
                controller.last_event.metadata["objects"]
            )
            counters = [source, moved, destination]
            self.demo._move_mug_to_receptacle(controller, mug["objectId"], moved["objectId"])

            coverages: list[CandidateCoverage] = []
            for index, counter in enumerate(counters, start=1):
                poses = self.demo._interactable_poses(controller, counter["objectId"])
                sampled = _evenly_spaced(poses, self.views_per_candidate)
                images: list[str] = []
                visible_count = 0
                for view_index, pose in enumerate(sampled, start=1):
                    self.demo._teleport(controller, pose)
                    event = controller.last_event
                    current_mug = next(
                        obj for obj in event.metadata["objects"] if obj["objectId"] == mug["objectId"]
                    )
                    visible_count += int(bool(current_mug.get("visible")))
                    frame = self.demo._save_frame(
                        event.frame,
                        f"candidate_{index}_view_{view_index}_visible_{int(bool(current_mug.get('visible')))}.png",
                    )
                    images.append(str(frame))
                coverages.append(
                    CandidateCoverage(
                        label=f"counter_{index}",
                        object_id=counter["objectId"],
                        sampled_views=len(sampled),
                        mug_visible_views=visible_count,
                        image_paths=tuple(images),
                    )
                )

            report_path = self.output_dir / "coverage-report.json"
            payload = {
                "scene": self.demo.scene,
                "moved_counter_id": moved["objectId"],
                "candidates": [asdict(coverage) for coverage in coverages],
            }
            report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return CoverageReport(
                scene=self.demo.scene,
                moved_counter_id=moved["objectId"],
                candidates=tuple(coverages),
                report_path=str(report_path),
            )
        finally:
            controller.stop()


def _evenly_spaced(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(values) <= count:
        return values
    indices = {round(index * (len(values) - 1) / (count - 1)) for index in range(count)}
    return [value for index, value in enumerate(values) if index in indices]


def main() -> None:
    report = ThorVisualCoverageDiagnostics().run()
    print(f"scene={report.scene}")
    print(f"moved_counter={report.moved_counter_id}")
    for candidate in report.candidates:
        print(
            f"{candidate.label}: visible_mug_views={candidate.mug_visible_views}/{candidate.sampled_views} "
            f"object_id={candidate.object_id}"
        )
    print(f"report={Path(report.report_path).resolve()}")


if __name__ == "__main__":
    main()
