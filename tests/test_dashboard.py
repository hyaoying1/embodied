import tempfile
import unittest
from pathlib import Path

from fast_slow_agent.dashboard import render_dashboard, write_dashboard


class DashboardTests(unittest.TestCase):
    def test_dashboard_renders_comparison_and_delegation_trace(self) -> None:
        dashboard = render_dashboard()
        self.assertIn("Fast–Slow Embodied Agent", dashboard)
        self.assertIn("Always Fast with target move", dashboard)
        self.assertIn("Fast–Slow Router with target move", dashboard)
        self.assertIn("Memory conflicts with the inspected cabinet", dashboard)

    def test_dashboard_writes_self_contained_html_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = write_dashboard(Path(directory) / "demo.html")
            self.assertTrue(output.exists())
            self.assertIn("<!doctype html>", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
