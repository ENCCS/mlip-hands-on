#!/usr/bin/env python3
"""Make identical silicon positions and velocities for both MD engines.

The JSON stores ALCHEMI's internal velocity units (Å per 10.1805057 fs).
LAMMPS metal-unit data stores the same velocities in Å/ps.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


LATTICE_A = 5.43
MASS_AMU = 28.085
KB_EV_PER_K = 8.617333262145e-5
FS_PER_INTERNAL_TIME = 10.180505710759414  # nvalchemi-toolkit 0.2.0
BASIS = (
    (0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
    (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75), (0.75, 0.75, 0.25),
)


def make_state(cells, replica, temperature):
    positions = [
        [LATTICE_A * (index[axis] + site[axis]) for axis in range(3)]
        for i in range(cells) for j in range(cells) for k in range(cells)
        for index in [(i, j, k)] for site in BASIS
    ]
    rng = random.Random(20260930 + replica)
    sigma = math.sqrt(KB_EV_PER_K * temperature / MASS_AMU)
    velocities = [[rng.gauss(0.0, sigma) for _ in range(3)]
                  for _ in positions]
    mean = [sum(v[axis] for v in velocities) / len(velocities)
            for axis in range(3)]
    velocities = [[v[axis] - mean[axis] for axis in range(3)]
                  for v in velocities]
    return positions, velocities


def write_state(directory, cells, replica, temperature):
    positions, velocities = make_state(cells, replica, temperature)
    side = cells * LATTICE_A
    record = {
        "format": 1, "atoms": len(positions), "cells": cells,
        "replica": replica, "temperature_initial_k": temperature,
        "cell_a": side, "velocity_unit": "angstrom/internal-time",
        "positions_a": positions, "velocities_internal": velocities,
    }
    json_path = directory / f"replica-{replica}.json"
    data_path = directory / f"replica-{replica}.data"
    json_path.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    with data_path.open("w") as stream:
        stream.write("Matched silicon NVE start\n\n")
        stream.write(f"{len(positions)} atoms\n1 atom types\n\n")
        for axis in "xyz":
            stream.write(f"0 {side:.16g} {axis}lo {axis}hi\n")
        stream.write("\nMasses\n\n1 28.085\n\nAtoms # atomic\n\n")
        for atom_id, position in enumerate(positions, 1):
            stream.write(f"{atom_id} 1 " + " ".join(f"{v:.16g}" for v in position) + "\n")
        stream.write("\nVelocities\n\n")
        for atom_id, velocity in enumerate(velocities, 1):
            metal = [v * 1000.0 / FS_PER_INTERNAL_TIME for v in velocity]
            stream.write(f"{atom_id} " + " ".join(f"{v:.16g}" for v in metal) + "\n")
    return {
        "replica": replica,
        "json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.cells <= 30 and 1 <= args.replicas <= 64):
        parser.error("cells must be 1..30 and replicas 1..64")
    if 8 * args.cells**3 * args.replicas > 500_000:
        parser.error("total starting atoms must not exceed 500,000")
    if not math.isfinite(args.temperature_k) or args.temperature_k <= 0:
        parser.error("temperature must be finite and positive")
    args.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    identities = [write_state(args.output_dir, args.cells, i, args.temperature_k)
                  for i in range(args.replicas)]
    manifest = {
        "format": 1, "cells": args.cells, "atoms_per_replica": 8 * args.cells**3,
        "replicas": args.replicas, "temperature_initial_k": args.temperature_k,
        "timestep_fs": 0.1, "ensemble": "NVE", "starts": identities,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {args.replicas} matched starts with {8 * args.cells**3} atoms each")


if __name__ == "__main__":
    main()
