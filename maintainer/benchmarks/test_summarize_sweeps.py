"""Matrix completeness, per-replica completion, and final-value regression tests."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parent


def log_text(atoms, ranks=1, finite=True):
    energy = "-300" if finite else "nan"
    return (f"  will use up to {ranks} GPU(s) per node\n"
            f"Loop time of 0.1 on {ranks} procs for 0 steps with {atoms} atoms\n"
            f"Loop time of 1.0 on {ranks} procs for 10 steps with {atoms} atoms\n"
            f"210 {atoms} 300 {energy} 1 -299\n"
            f"Loop time of 5.0 on {ranks} procs for 200 steps with {atoms} atoms\n"
            "Total wall time: 0:00:30\n")


class SweepTests(unittest.TestCase):
    def make_sweep(self, root):
        rows = []
        for round_no in (1, 2, 3):
            for cells, replicas in ((2, 1), (4, 1), (10, 1), (16, 1), (2, 2), (4, 2), (2, 4), (4, 4)):
                atoms = 8 * cells**3
                for method in ("alchemi", "ordinary", "mps"):
                    if method == "mps" and replicas == 1:
                        continue
                    rows.append([round_no, cells, atoms, method, replicas, 10, 200, 30, 5])
                    stem = f"round{round_no}-atoms{atoms}-r{replicas}-{method}"
                    record = {"state": "completed", "ensemble": "NVE", "replicas": replicas,
                              "atoms_per_replica": atoms, "warmup_steps": 10, "measured_steps": 200}
                    if method == "alchemi":
                        record.update(md_seconds=5.0, final_potential_ev=[-300.0] * replicas)
                        (root / f"{stem}.log").write_text(json.dumps(record, separators=(",", ":")))
                    else:
                        record.update(replica_md_loop_seconds=[5.0] * replicas)
                        (root / f"{stem}-summary.json").write_text(json.dumps(record))
                        logs = root / f"{stem}-replicas"
                        logs.mkdir()
                        for index in range(replicas):
                            (logs / f"replica-{index}.log").write_text(log_text(atoms))
                        if method == "mps":
                            (root / f"{stem}-mps-proof").write_text("observed\n")
        with (root / "results.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["round", "cells", "atoms", "method", "replicas", "warmup", "steps", "whole_seconds", "md_diagnostic_seconds"])
            writer.writerows(rows)

    def run_sweep(self, root):
        return subprocess.run([sys.executable, str(SOURCE / "summarize_sweep.py"),
                               "--results", str(root), "--phase", "all", "--output", str(root / "accepted.json")],
                              capture_output=True, text=True)

    def test_complete_sweep_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_sweep(root)
            outcome = self.run_sweep(root)
            self.assertEqual(outcome.returncode, 0, outcome.stderr)

            self.assertEqual(json.loads((root / "accepted.json").read_text())["rows_checked"], 60)

    def test_missing_replica_cannot_be_hidden_by_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_sweep(root)
            (root / "round3-atoms512-r4-ordinary-replicas/replica-3.log").unlink()
            self.assertNotEqual(self.run_sweep(root).returncode, 0)
            self.assertFalse((root / "accepted.json").exists())

    def test_nonfinite_final_thermo_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_sweep(root)
            (root / "round3-atoms512-r4-ordinary-replicas/replica-3.log").write_text(log_text(512, finite=False))
            self.assertNotEqual(self.run_sweep(root).returncode, 0)
            self.assertFalse((root / "accepted.json").exists())

    def test_mps_proof_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_sweep(root)
            (root / "round1-atoms64-r2-mps-mps-proof").unlink()
            self.assertNotEqual(self.run_sweep(root).returncode, 0)

    def test_scaling_requires_complete_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "results.tsv").write_text("round\tscaling\tranks\tatoms\twarmup\tsteps\tmd_seconds\n")
            outcome = subprocess.run([sys.executable, str(SOURCE / "summarize_scaling_matrix.py"),
                                      "--results", str(root), "--output", str(root / "accepted.json")],
                                     capture_output=True, text=True)
            self.assertNotEqual(outcome.returncode, 0)
            self.assertFalse((root / "accepted.json").exists())

    def test_complete_scaling_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for round_no in (1, 2, 3):
                for mode in ("strong", "weak"):
                    for ranks in (1, 2, 4):
                        atoms = 32768 if mode == "strong" else {1: 8000, 2: 17576, 4: 32768}[ranks]
                        stem = f"round{round_no}-{mode}-g{ranks}-a{atoms}"
                        (root / f"{stem}.log").write_text(log_text(atoms, ranks))
                        (root / f"{stem}-summary.json").write_text(json.dumps({
                            "state": "completed", "ranks": ranks, "atoms": atoms, "ensemble": "NVE",
                            "warmup_steps": 10, "measured_steps": 200, "md_seconds": 5.0}))
                        rows.append([round_no, mode, ranks, atoms, 10, 200, 5.0])
            with (root / "results.tsv").open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["round", "scaling", "ranks", "atoms", "warmup", "steps", "md_seconds"])
                writer.writerows(rows)
            outcome = subprocess.run([sys.executable, str(SOURCE / "summarize_scaling_matrix.py"),
                                      "--results", str(root), "--output", str(root / "accepted.json")],
                                     capture_output=True, text=True)
            self.assertEqual(outcome.returncode, 0, outcome.stderr)

            # A matching MPI count is not enough if Kokkos reports fewer GPUs.
            log = root / "round3-strong-g4-a32768.log"
            log.write_text(log.read_text().replace("4 GPU(s)", "1 GPU(s)"))
            rejected = subprocess.run([sys.executable, str(SOURCE / "summarize_scaling_matrix.py"),
                                       "--results", str(root), "--output", str(root / "bad-summary.json")],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / "bad-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
