#!/usr/bin/env python3
"""Time independent 64-atom Si/MACE trajectories in native LAMMPS."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from ase.build import bulk


def one_replica(model, steps, warmup, seed, integrator, cells,
                atom_sort_interval):
    import lammps
    from lammps.mliap import activate_mliappy_kokkos

    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((cells, cells, cells))
    count = len(atoms)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-k", "on", "g", "1"])
    try:
        if not lmp.has_package("KOKKOS") or not lmp.has_style("pair", "mliap/kk"):
            raise RuntimeError("LAMMPS needs KOKKOS and mliap/kk")
        if not lmp.has_style("fix", "nve/kk"):
            raise RuntimeError("LAMMPS needs nve/kk")
        activate_mliappy_kokkos(lmp)
        length = float(atoms.cell.lengths()[0])
        for command in (
            "package kokkos newton on neigh half",
            "units metal", "atom_style atomic/kk", "newton on", "boundary p p p",
            f"region box block 0 {length} 0 {length} 0 {length}",
            "create_box 1 box", "run_style verlet/kk", "mass 1 28.085",
        ):
            lmp.command(command)
        if count == 64:
            for position in np.asarray(atoms.positions):
                lmp.command("create_atoms 1 single " + " ".join(f"{v:.17g}" for v in position) + " units box")
        else:
            positions = np.asarray(atoms.positions, dtype=np.float64).reshape(-1).tolist()
            created = lmp.create_atoms(count, None, [1] * count, positions)
            if created != count:
                raise RuntimeError("LAMMPS bulk atom creation failed")
        if lmp.get_natoms() != count:
            raise RuntimeError("LAMMPS atom count mismatch")
        for command in (
            f"atom_modify sort {atom_sort_interval} 0.0",
            f"pair_style mliap/kk unified {model} 0",
            "pair_coeff * * Si",
            f"velocity all create 300 {seed} mom yes rot yes dist gaussian",
            "timestep 0.0001",  # LAMMPS metal time is ps: 0.1 fs
            "fix integrate all nve/kk",
        ):
            lmp.command(command)
        if integrator == "langevin":
            lmp.command(f"fix bath all langevin 300 300 0.1 {seed + 1}")
        lmp.command("run 0")
        if warmup:
            lmp.command(f"run {warmup}")
        started = time.perf_counter()
        lmp.command(f"run {steps}")
        elapsed = time.perf_counter() - started
        energy = float(lmp.get_thermo("pe"))
        if not math.isfinite(energy):
            raise RuntimeError("non-finite potential energy")
        return elapsed, energy
    finally:
        lmp.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mliap-model", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--integrator", choices=("langevin", "nve"), default="langevin")
    parser.add_argument("--cells", type=int, choices=(2, 4, 8, 10, 16, 17, 18), default=2)
    parser.add_argument("--replica-start", type=int, default=0,
                        help="first global replica index for disjoint process seeds")
    parser.add_argument("--atom-sort-interval", type=int, choices=(0, 100), default=0,
                        help="0 preserves the baseline; 100 tests documented GPU sorting")
    args = parser.parse_args()
    if not args.mliap_model.is_file():
        parser.error("ML-IAP export is absent")
    if not 1 <= args.replicas <= 64 or not 1 <= args.steps <= 10000:
        parser.error("replicas must be 1..64 and steps 1..10000")
    if not 0 <= args.warmup <= 1000:
        parser.error("warmup must be 0..1000")
    if not 0 <= args.replica_start <= 63 or args.replica_start + args.replicas > 64:
        parser.error("replica indices must be within 0..63")
    atoms_per_replica = 8 * args.cells**3
    if atoms_per_replica * args.replicas > 64000:
        parser.error("unreviewed total atom count")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("exactly one visible CUDA GPU is required")

    digest = hashlib.sha256()
    with args.mliap_model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    total_started = time.perf_counter()
    measured = []
    final_energies = []
    for index in range(args.replicas):
        elapsed, energy = one_replica(args.mliap_model, args.steps, args.warmup,
                                      20260924 + 2 * (args.replica_start + index),
                                      args.integrator, args.cells,
                                      args.atom_sort_interval)
        measured.append(elapsed)
        final_energies.append(energy)
    total_seconds = time.perf_counter() - total_started
    measured_seconds = sum(measured)
    print(json.dumps({
        "format": 1, "engine": "lammps-mliap-kokkos", "state": "completed",
        "integrator": args.integrator,
        "replicas": args.replicas, "atoms_per_replica": atoms_per_replica,
        "replica_start": args.replica_start,
        "warmup_steps": args.warmup, "measured_steps": args.steps,
        "timestep_fs": 0.1, "temperature_k": 300.0,
        "total_seconds": total_seconds, "measured_seconds": measured_seconds,
        "replica_steps_per_second": args.replicas * args.steps / measured_seconds,
        "end_to_end_replica_steps_per_second": args.replicas * args.steps / total_seconds,
        "mean_replica_latency_seconds": measured_seconds / args.replicas,
        "atom_sort_interval": args.atom_sort_interval,
        "final_potential_energies_ev": final_energies,
        "local_rank": int(os.environ.get("SLURM_LOCALID", "0")),
        "mliap_model_sha256": digest.hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
