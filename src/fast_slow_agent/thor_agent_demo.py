"""Run the shared FastSlowAgent loop against the AI2-THOR adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .agent import FastSlowAgent
from .ai2thor_environment import AI2ThorEnvironment
from .interfaces import SlowPlanner
from .models import EpisodeResult
from .thor_demo import AI2ThorFastSlowDemo


@dataclass(frozen=True)
class ThorAgentLoopResult:
    """Result and stable area IDs for a shared-agent THOR episode."""

    episode: EpisodeResult
    source_counter_id: str
    moved_counter_id: str
    destination_counter_id: str

    @property
    def success(self) -> bool:
        return self.episode.success


class AI2ThorAgentDemo:
    """Construct an adapter and run the normal FastSlowAgent control loop.

    This entry point is intentionally separate from ``AI2ThorFastSlowDemo``:
    the latter remains a pre-scripted oracle/VLM validation harness, whereas
    this class proves that the shared high-level agent can drive AI2-THOR.
    """

    def __init__(
        self,
        *,
        scene: str = "FloorPlan1",
        controller_factory: Callable[[], Any] | None = None,
        slow_planner: SlowPlanner | None = None,
    ) -> None:
        self.scene = scene
        self.controller_factory = controller_factory
        self.slow_planner = slow_planner

    def run(self, *, max_steps: int = 12) -> ThorAgentLoopResult:
        bridge = AI2ThorFastSlowDemo(scene=self.scene)
        controller = self.controller_factory() if self.controller_factory else bridge._create_controller()
        try:
            mug, source, moved, destination = bridge._discover_objects(
                controller.last_event.metadata["objects"]
            )
            environment = AI2ThorEnvironment(
                controller=controller,
                target_object_id=mug["objectId"],
                initial_target_location=source["objectId"],
                destination_location=destination["objectId"],
                locations={
                    source["objectId"]: source["objectId"],
                    moved["objectId"]: moved["objectId"],
                    destination["objectId"]: destination["objectId"],
                },
                navigation_object_ids={
                    source["objectId"]: mug["objectId"],
                    moved["objectId"]: mug["objectId"],
                    destination["objectId"]: destination["objectId"],
                },
                dynamic_event=True,
                moved_target_location=moved["objectId"],
            )
            episode = FastSlowAgent(slow_planner=self.slow_planner).run(
                environment,
                max_steps=max_steps,
            )
            return ThorAgentLoopResult(
                episode=episode,
                source_counter_id=source["objectId"],
                moved_counter_id=moved["objectId"],
                destination_counter_id=destination["objectId"],
            )
        finally:
            controller.stop()


def main() -> None:
    result = AI2ThorAgentDemo().run()
    print("AI2-THOR shared FastSlowAgent episode")
    print(f"source={result.source_counter_id}")
    print(f"moved_to={result.moved_counter_id}")
    print(f"destination={result.destination_counter_id}")
    print(f"success={result.success}")
    print(f"slow_calls={result.episode.slow_calls}")
    if result.episode.failure_reason:
        print(f"failure_reason={result.episode.failure_reason}")


if __name__ == "__main__":
    main()
