"""Terminal visualization for the fast–slow delegation MVP."""

from __future__ import annotations

from .evaluation import comparison_suite
from .models import EpisodeResult


def render_result(label: str, result: EpisodeResult) -> str:
    lines = [
        "=" * 88,
        label,
        f"agent={result.agent_name} | success={result.success} | steps={result.steps} | "
        f"slow_calls={result.slow_calls} | dynamic_event={result.event_triggered}",
        "-" * 88,
    ]
    for entry in result.trace:
        lines.extend(
            [
                f"[step {entry.step:02d}] {entry.action:<30} success={entry.success}",
                f"  mode={entry.mode:<10} confidence={entry.confidence:.2f} slow_calls={entry.slow_calls}",
                f"  reason={entry.reason}",
                f"  memory={entry.memory_summary}",
                f"  event/result={entry.message}",
            ]
        )
    if result.failure_reason:
        lines.append(f"FINAL: FAILED — {result.failure_reason}")
    else:
        lines.append("FINAL: SUCCESS — blue_mug reached dining_table.")
    return "\n".join(lines)


def main() -> None:
    print("Fast–Slow Embodied Agent MVP")
    print("Fast Brain uses transparent routing rules; Slow Brain is explicitly an oracle planner.\n")
    for label, result in comparison_suite():
        print(render_result(label, result))
    print("=" * 88)
    print("Expected comparison: Always Fast fails after the target moves; Fast–Slow delegates once and recovers.")


if __name__ == "__main__":
    main()
