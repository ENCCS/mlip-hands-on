#!/usr/bin/env python3
"""Check every completed LAMMPS replica and extract its measured loop time."""

import argparse
import json
import math
import os
import re
from pathlib import Path


LOOP = re.compile(r"Loop time of ([0-9.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicas < 1 or args.atoms < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("invalid workload shape")
    times = []
    for index in range(args.replicas):
        content = (args.logs / f"replica-{index}.log").read_text()
        if content.count("Total wall time:") != 1 or "ERROR:" in content:
            raise RuntimeError(f"replica {index} did not complete cleanly")
        loops = [(float(seconds), int(steps), int(atoms))
                 for seconds, steps, atoms in LOOP.findall(content)]
        if (len(loops) != 3 or [row[1] for row in loops] !=
                [0, args.warmup, args.steps] or
                any(row[2] != args.atoms for row in loops)):
            raise RuntimeError(f"replica {index} has incomplete MD loops")
        seconds = loops[-1][0]
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError(f"replica {index} has invalid measured time")
        times.append(seconds)
    result = {
        "state": "completed", "engine": "lammps", "ensemble": "NVE",
        "replicas": args.replicas, "atoms_per_replica": args.atoms,
        "warmup_steps": args.warmup, "measured_steps": args.steps,
        "replica_md_loop_seconds": times,
        "longest_replica_md_loop_seconds": max(times),
        "clock_note": "Per-replica LAMMPS loops; not a synchronized batch MD clock",
    }
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
