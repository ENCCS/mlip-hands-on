"""Fail-closed tests for the LAMMPS measured-loop summary."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("summarize_lammps_nve.py")
LOG = """Loop time of 0.01 on 1 procs for 0 steps with 64 atoms
Loop time of 1.1 on 1 procs for 10 steps with 64 atoms
Loop time of 2.5 on 1 procs for 200 steps with 64 atoms
Total wall time: 0:00:10
"""


class SummaryTests(unittest.TestCase):
    def invoke(self, root):
        return subprocess.run(
            [sys.executable, str(SOURCE), "--logs", str(root), "--replicas", "2",
             "--atoms", "64", "--warmup", "10", "--steps", "200",
             "--output", str(root / "summary.json")],
            capture_output=True, text=True,
        )

    def test_requires_every_replica(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "replica-0.log").write_text(LOG)
            self.assertNotEqual(self.invoke(root).returncode, 0)
            self.assertFalse((root / "summary.json").exists())

    def test_only_matching_three_loops_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "replica-0.log").write_text(LOG)
            (root / "replica-1.log").write_text(LOG)
            self.assertEqual(self.invoke(root).returncode, 0)
            result = json.loads((root / "summary.json").read_text())
            self.assertEqual(result["replica_md_loop_seconds"], [2.5, 2.5])
            self.assertIn("not a synchronized", result["clock_note"])


if __name__ == "__main__":
    unittest.main()
