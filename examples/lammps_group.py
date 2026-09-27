#!/usr/bin/env python3
"""Run eight independent LAMMPS replicas, sequentially or sharing one GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sequential", "plain", "mps"), required=True)
    parser.add_argument("--cells", type=int, choices=(2, 4, 8, 10), default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--integrator", choices=("nve", "langevin"), default="nve")
    args = parser.parse_args()
    runner = Path(__file__).resolve().parents[1] / "scripts" / "run-lammps.sh"
    commands = [
        ["bash", str(runner), "--replicas", "1", "--replica-start", str(i),
         "--cells", str(args.cells), "--steps", str(args.steps),
         "--warmup", str(args.warmup), "--integrator", args.integrator]
        for i in range(8)
    ]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1 if args.mode == "sequential" else 8) as pool:
        completed = list(pool.map(
            lambda command: subprocess.run(command, capture_output=True, text=True),
            commands,
        ))
    if any(result.returncode for result in completed):
        raise RuntimeError("A LAMMPS client failed; inspect private job diagnostics")
    rows = [json.loads(result.stdout.strip().splitlines()[-1])
            for result in completed]
    elapsed = time.perf_counter() - started
    assert len(rows) == 8 and all(r["state"] == "completed" for r in rows)
    print(json.dumps({
        "mode": args.mode,
        "replicas": 8, "atoms_per_replica": rows[0]["atoms_per_replica"],
        "steps": args.steps, "group_wall_seconds": elapsed,
        "replica_steps_per_second": 8 * args.steps / elapsed,
    }))


if __name__ == "__main__":
    main()
