"""Run the shared FastSlowAgent loop against the AI2-THOR adapter."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import FastSlowAgent
from .ai2thor_environment import AI2ThorEnvironment
from .interfaces import SlowPlanner
from .models import EpisodeResult
from .thor_demo import AI2ThorFastSlowDemo, TASK_DESCRIPTION
from .thor_vision_provider import AI2ThorVisionCandidateProvider
from .vision_planner import VisionSearchSlowPlanner
from .vision_slow_brain import OpenAICompatibleVisionClient, VisionSearchSlowBrain


@dataclass(frozen=True)
class ThorAgentLoopResult:
    episode: EpisodeResult
    source_counter_id: str
    moved_counter_id: str
    destination_counter_id: str

    @property
    def success(self) -> bool:
        return self.episode.success


class AI2ThorAgentDemo:
    """Run the shared agent loop on AI2-THOR with oracle or VLM planning."""

    def __init__(self, *, scene: str = "FloorPlan1", controller_factory: Callable[[], Any] | None = None, slow_planner: SlowPlanner | None = None) -> None:
        self.scene = scene
        self.controller_factory = controller_factory
        self.slow_planner = slow_planner

    @classmethod
    def with_vlm_from_environment(cls, *, output_dir: Path = Path("artifacts/thor/shared-vlm"), **kwargs: Any) -> "AI2ThorAgentDemo":
        client = OpenAICompatibleVisionClient.from_environment()
        planner = VisionSearchSlowPlanner(
            task=TASK_DESCRIPTION,
            candidate_provider=AI2ThorVisionCandidateProvider(output_dir=output_dir),
            vision_search=VisionSearchSlowBrain(client),
        )
        return cls(slow_planner=planner, **kwargs)

    def run(self, *, max_steps: int = 12) -> ThorAgentLoopResult:
        bridge = AI2ThorFastSlowDemo(scene=self.scene)
        controller = self.controller_factory() if self.controller_factory else bridge._create_controller()
        try:
            mug, source, moved, destination = bridge._discover_objects(controller.last_event.metadata["objects"])
            environment = AI2ThorEnvironment(
                controller=controller, target_object_id=mug["objectId"], initial_target_location=source["objectId"], destination_location=destination["objectId"],
                locations={source["objectId"]: source["objectId"], moved["objectId"]: moved["objectId"], destination["objectId"]: destination["objectId"]},
                navigation_object_ids={source["objectId"]: mug["objectId"], moved["objectId"]: mug["objectId"], destination["objectId"]: destination["objectId"]},
                dynamic_event=True, moved_target_location=moved["objectId"],
            )
            episode = FastSlowAgent(slow_planner=self.slow_planner).run(environment, max_steps=max_steps)
            return ThorAgentLoopResult(episode, source["objectId"], moved["objectId"], destination["objectId"])
        finally:
            controller.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared FastSlowAgent on AI2-THOR.")
    parser.add_argument("--slow-brain", choices=("oracle", "vlm"), default="oracle")
    args = parser.parse_args()
    demo = AI2ThorAgentDemo.with_vlm_from_environment() if args.slow_brain == "vlm" else AI2ThorAgentDemo()
    result = demo.run()
    print("AI2-THOR shared FastSlowAgent episode")
    print(f"slow_brain={args.slow_brain}\nsuccess={result.success}\nslow_calls={result.episode.slow_calls}")
    if result.episode.failure_reason:
        print(f"failure_reason={result.episode.failure_reason}")


if __name__ == "__main__":
    main()
