"""CPU-only check for complete MPI timing recognition."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("summarize_lammps_nve_mpi.py")
LOG = """Loop time of 0.2 on 2 procs for 0 steps with 32768 atoms
Loop time of 3.1 on 2 procs for 10 steps with 32768 atoms
Loop time of 42.7 on 2 procs for 100 steps with 32768 atoms
Total wall time: 0:00:50
"""


class MpiSummaryTests(unittest.TestCase):
    def invoke(self, root, ranks):
        return subprocess.run(
            [sys.executable, str(SOURCE), "--log", str(root / "case.log"),
             "--ranks", str(ranks), "--atoms", "32768", "--warmup", "10",
             "--steps", "100", "--output", str(root / "result.json")],
            capture_output=True, text=True,
        )

    def test_exact_rank_and_steps_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "case.log").write_text(LOG)
            self.assertNotEqual(self.invoke(root, 4).returncode, 0)
            self.assertFalse((root / "result.json").exists())
            self.assertEqual(self.invoke(root, 2).returncode, 0)
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["md_seconds"], 42.7)


if __name__ == "__main__":
    unittest.main()
