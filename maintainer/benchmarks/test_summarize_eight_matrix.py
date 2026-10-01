"""The whole matrix must be present before a result is accepted."""

import subprocess
import json
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("summarize_eight_matrix.py")


class MatrixTests(unittest.TestCase):
    def complete_fixture(self, root):
        rows = ["round\tcells\tatoms\tmethod\treplicas\twarmup\tsteps\twhole_seconds\tmd_diagnostic_seconds"]
        for round_no in (1, 2, 3):
            for cells in (2, 4):
                atoms = 8 * cells**3
                for method in ("alchemi", "ordinary", "mps"):
                    rows.append(f"{round_no}\t{cells}\t{atoms}\t{method}\t8\t10\t200\t30.0\t5.0")
                    stem = f"round{round_no}-atoms{atoms}-r8-{method}"
                    record = {"state": "completed", "ensemble": "NVE", "replicas": 8,
                              "atoms_per_replica": atoms, "warmup_steps": 10,
                              "measured_steps": 200}
                    if method == "alchemi":
                        record.update(md_seconds=5.0, final_potential_ev=[-300.0] * 8)
                        (root / f"{stem}.log").write_text(json.dumps(record, separators=(",", ":")))
                    else:
                        record.update(replica_md_loop_seconds=[5.0] * 8,
                                      longest_replica_md_loop_seconds=5.0)
                        (root / f"{stem}-summary.json").write_text(json.dumps(record))
                        (root / f"{stem}.log").write_text("completed launcher\n")
                        logs = root / f"{stem}-replicas"
                        logs.mkdir()
                        for replica in range(8):
                            (logs / f"replica-{replica}.log").write_text(
                                f"Loop time of 0.1 on 1 procs for 0 steps with {atoms} atoms\n"
                                f"Loop time of 1.0 on 1 procs for 10 steps with {atoms} atoms\n"
                                f"210 {atoms} 300 -300 1 -299\n"
                                f"Loop time of 5.0 on 1 procs for 200 steps with {atoms} atoms\n"
                                "Total wall time: 0:00:30\n")
        (root / "results.tsv").write_text("\n".join(rows) + "\n")

    def run_summary(self, root):
        return subprocess.run([sys.executable, str(SOURCE), "--results", str(root),
                               "--output", str(root / "accepted.json")],
                              capture_output=True, text=True)

    def test_complete_current_stems_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.complete_fixture(root)
            result = self.run_summary(root)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_nonfinite_energy_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.complete_fixture(root)
            path = root / "round1-atoms64-r8-alchemi.log"
            record = json.loads(path.read_text())
            record["final_potential_ev"][0] = float("nan")
            path.write_text(json.dumps(record, separators=(",", ":")))
            self.assertNotEqual(self.run_summary(root).returncode, 0)
            self.assertFalse((root / "accepted.json").exists())

    def test_wrong_ensemble_refuses(self):
        for suffix in ("alchemi.log", "ordinary-summary.json"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.complete_fixture(root)
                path = root / f"round1-atoms64-r8-{suffix}"
                record = json.loads(path.read_text())
                record["ensemble"] = "NVT"
                path.write_text(json.dumps(record, separators=(",", ":")))
                self.assertNotEqual(self.run_summary(root).returncode, 0)
                self.assertFalse((root / "accepted.json").exists())

    def test_md_interval_cannot_exceed_whole_workflow(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.complete_fixture(root)
            table = root / "results.tsv"
            table.write_text(table.read_text().replace("30.0\t5.0", "1.0\t5.0", 1))
            self.assertNotEqual(self.run_summary(root).returncode, 0)
            self.assertFalse((root / "accepted.json").exists())

    def test_separate_mps_phase_is_not_a_cross_engine_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.complete_fixture(root)
            path = root / "results.tsv"
            lines = path.read_text().splitlines()
            path.write_text("\n".join([lines[0]] + [line for line in lines[1:] if "\tmps\t" in line]) + "\n")
            result = subprocess.run([sys.executable, str(SOURCE), "--results", str(root),
                                     "--phase", "mps", "--output", str(root / "accepted.json")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / "accepted.json").read_text())
            self.assertEqual(summary["phase"], "mps")
            self.assertEqual(set(summary["whole_workflow_medians"]["64"]), {"mps"})

    def test_summary_cannot_hide_failed_replica(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.complete_fixture(root)
            (root / "round1-atoms64-r8-mps-replicas/replica-7.log").write_text("ERROR: failed\n")
            self.assertNotEqual(self.run_summary(root).returncode, 0)
            self.assertFalse((root / "accepted.json").exists())

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
