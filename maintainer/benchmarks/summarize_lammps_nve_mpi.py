#!/usr/bin/env python3
"""Check a complete one-trajectory LAMMPS MPI timing log."""

import argparse
import json
import math
import os
import re
from pathlib import Path


LOOP = re.compile(r"Loop time of ([0-9.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ranks", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.log.read_text()
    if content.count("Total wall time:") != 1 or "ERROR:" in content:
        raise RuntimeError("LAMMPS MPI run did not complete cleanly")
    loops = [(float(seconds), int(ranks), int(steps), int(atoms))
             for seconds, ranks, steps, atoms in LOOP.findall(content)]
    if (len(loops) != 3 or [row[2] for row in loops] !=
            [0, args.warmup, args.steps] or
            any(row[1] != args.ranks or row[3] != args.atoms for row in loops)):
        raise RuntimeError("LAMMPS MPI loops do not match the declared workload")
    measured_seconds = loops[-1][0]
    if not math.isfinite(measured_seconds) or measured_seconds <= 0:
        raise RuntimeError("invalid measured loop time")
    result = {"state": "completed", "engine": "lammps", "ensemble": "NVE",
              "ranks": args.ranks, "gpus": args.ranks, "atoms": args.atoms,
              "warmup_steps": args.warmup, "measured_steps": args.steps,
              "md_seconds": measured_seconds}
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
