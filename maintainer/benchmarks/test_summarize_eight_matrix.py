"""The whole matrix must be present before a result is accepted."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("summarize_eight_matrix.py")


class MatrixTests(unittest.TestCase):
    def test_partial_matrix_refuses_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "results.tsv").write_text(
                "round\tcells\tatoms\tmethod\treplicas\twarmup\tsteps\twhole_seconds\tmd_diagnostic_seconds\n"
                "1\t2\t64\talchemi\t8\t10\t200\t30.0\t5.0\n"
            )
            outcome = subprocess.run(
                [sys.executable, str(SOURCE), "--results", str(root),
                 "--output", str(root / "accepted.json")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(outcome.returncode, 0)
            self.assertFalse((root / "accepted.json").exists())


if __name__ == "__main__":
    unittest.main()
