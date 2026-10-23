"""Generate a self-contained browser dashboard for the comparison demo."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .evaluation import comparison_suite
from .models import EpisodeResult, TraceEntry


DEFAULT_OUTPUT = Path("artifacts/fast-slow-demo-dashboard.html")


def _status_class(result: EpisodeResult) -> str:
    return "success" if result.success else "failure"


def _render_summary_row(label: str, result: EpisodeResult) -> str:
    status = "SUCCESS" if result.success else "FAILED"
    return f"""
      <tr>
        <td>{escape(label)}</td>
        <td>{escape(result.agent_name)}</td>
        <td><span class=\"badge {_status_class(result)}\">{status}</span></td>
        <td>{result.steps}</td>
        <td>{result.slow_calls}</td>
        <td>{'yes' if result.event_triggered else 'no'}</td>
      </tr>
    """


def _render_trace(entry: TraceEntry) -> str:
    result_class = "ok" if entry.success else "error"
    return f"""
      <article class=\"trace {result_class}\">
        <div class=\"trace-heading\">
          <strong>Step {entry.step:02d}: {escape(entry.action)}</strong>
          <span>{escape(entry.mode)} · confidence {entry.confidence:.2f} · slow calls {entry.slow_calls}</span>
        </div>
        <p><b>Reason:</b> {escape(entry.reason)}</p>
        <p><b>Memory:</b> {escape(entry.memory_summary)}</p>
        <p><b>Event / result:</b> {escape(entry.message)}</p>
      </article>
    """


def _render_episode(label: str, result: EpisodeResult) -> str:
    final = "Task completed" if result.success else escape(result.failure_reason or "Task failed")
    trace = "".join(_render_trace(entry) for entry in result.trace)
    return f"""
      <section class=\"episode {_status_class(result)}\">
        <div class=\"episode-header\">
          <div>
            <p class=\"eyebrow\">{escape(label)}</p>
            <h2>{escape(result.agent_name)}</h2>
          </div>
          <span class=\"badge {_status_class(result)}\">{'SUCCESS' if result.success else 'FAILED'}</span>
        </div>
        <div class=\"stats\">
          <div><span>Environment changed</span><b>{'Yes' if result.event_triggered else 'No'}</b></div>
          <div><span>Environment steps</span><b>{result.steps}</b></div>
          <div><span>Slow Brain calls</span><b>{result.slow_calls}</b></div>
        </div>
        <p class=\"final\"><b>Final:</b> {final}</p>
        <details open>
          <summary>Decision trace</summary>
          {trace}
        </details>
      </section>
    """


def render_dashboard() -> str:
    """Return a self-contained HTML report for the current comparison suite."""
    results = comparison_suite()
    summary_rows = "".join(_render_summary_row(label, result) for label, result in results)
    episodes = "".join(_render_episode(label, result) for label, result in results)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Fast–Slow Embodied Agent Demo</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1020;
      --surface: #151d35;
      --surface-light: #202b4c;
      --text: #edf2ff;
      --muted: #aab7d7;
      --green: #52d6a6;
      --red: #ff7b89;
      --blue: #87b9ff;
      --line: #33446f;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top right, #1e3261, var(--bg) 45%); color: var(--text); font: 15px/1.55 system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 48px 24px 72px; }}
    h1 {{ margin: 4px 0 12px; font-size: clamp(2rem, 5vw, 3.7rem); line-height: 1.05; }}
    h2 {{ margin: 0; font-size: 1.35rem; }}
    p {{ margin: 0.55rem 0; }}
    .eyebrow {{ color: var(--blue); font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .76rem; }}
    .lede {{ max-width: 760px; color: var(--muted); font-size: 1.08rem; }}
    .callout {{ margin: 28px 0; padding: 18px 20px; border-left: 4px solid var(--blue); background: rgba(32,43,76,.75); border-radius: 0 12px 12px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 28px 0 42px; background: rgba(21,29,53,.9); border-radius: 14px; overflow: hidden; }}
    th, td {{ padding: 13px 15px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
    tr:last-child td {{ border-bottom: 0; }}
    .badge {{ border-radius: 999px; display: inline-block; padding: 3px 9px; font-size: .75rem; font-weight: 800; letter-spacing: .04em; }}
    .badge.success {{ color: #052116; background: var(--green); }}
    .badge.failure {{ color: #3b0911; background: var(--red); }}
    .episode {{ margin: 22px 0; padding: 24px; background: rgba(21,29,53,.94); border: 1px solid var(--line); border-radius: 16px; }}
    .episode.success {{ border-top: 4px solid var(--green); }}
    .episode.failure {{ border-top: 4px solid var(--red); }}
    .episode-header, .trace-heading {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }}
    .stats div {{ background: var(--surface-light); padding: 12px; border-radius: 10px; }}
    .stats span {{ display: block; color: var(--muted); font-size: .78rem; }}
    .stats b {{ font-size: 1.3rem; }}
    .final {{ padding: 12px; border-radius: 9px; background: rgba(11,16,32,.62); }}
    summary {{ cursor: pointer; color: var(--blue); font-weight: 700; margin-top: 18px; }}
    .trace {{ margin: 12px 0; padding: 14px; background: #10182d; border-radius: 10px; border-left: 3px solid var(--green); }}
    .trace.error {{ border-left-color: var(--red); }}
    .trace-heading span {{ color: var(--muted); font-size: .84rem; text-align: right; }}
    .trace p {{ color: var(--muted); font-size: .9rem; }}
    .trace b {{ color: var(--text); }}
    footer {{ margin-top: 38px; color: var(--muted); font-size: .86rem; }}
    @media (max-width: 680px) {{ .stats {{ grid-template-columns: 1fr; }} th:nth-child(2), td:nth-child(2) {{ display: none; }} .episode-header, .trace-heading {{ flex-direction: column; }} .trace-heading span {{ text-align: left; }} }}
  </style>
</head>
<body>
  <main>
    <p class=\"eyebrow\">Mechanism MVP · visual report</p>
    <h1>Fast–Slow Embodied Agent</h1>
    <p class=\"lede\">A fast decision layer handles ordinary actions. When a dynamic scene change contradicts its memory, it delegates once to a slower planner, updates the belief state, and recovers the task.</p>
    <div class=\"callout\"><b>Important:</b> the current Slow Brain is an explicitly labeled oracle planner used to validate the delegation loop. The Fast Brain's delegate decision is now made by a tiny locally trained logistic classifier on synthetic teacher labels; this report does not claim real VLM/LLM perception or 3D execution.</div>
    <h2>Scenario comparison</h2>
    <table>
      <thead><tr><th>Scenario</th><th>Agent</th><th>Outcome</th><th>Steps</th><th>Slow calls</th><th>Target moved</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
    <h2>Decision traces</h2>
    {episodes}
    <footer>Generated locally from deterministic demo trajectories. The 3D AI2-THOR backend is an optional next-stage integration.</footer>
  </main>
</body>
</html>
"""


def write_dashboard(output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dashboard(), encoding="utf-8")
    return output


def main() -> None:
    output = write_dashboard()
    print(f"Dashboard written to {output.resolve()}")


if __name__ == "__main__":
    main()
