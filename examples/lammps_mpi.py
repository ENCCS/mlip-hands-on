#!/usr/bin/env python3
"""One silicon/MACE trajectory in MPI-enabled LAMMPS ML-IAP/Kokkos.

Every MPI rank executes this file; each writes one private result. No mpi4py
is needed because the LAMMPS Python library uses MPI_COMM_WORLD implicitly.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


MODEL_SHA256 = "db578c556298ad3bb1f4a93faa50d540eb2b9792215e81ef7548dd7e20a746e8"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", type=int, choices=(4, 16), required=True)
    parser.add_argument("--steps", type=int, choices=(10, 100), required=True)
    args = parser.parse_args()
    if not args.model.is_file() or not args.output_dir.is_dir():
        parser.error("the pinned model and private output directory must exist")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(args.model)):
        parser.error("model path cannot be represented in a LAMMPS command")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        parser.error("model identity mismatch")
    return args


def run(args):
    import torch
    import lammps
    from lammps.mliap import activate_mliappy_kokkos

    expected = int(os.environ["SLURM_NTASKS"])
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    if nodes != 1 or expected not in (1, 2, 4) or not 0 <= rank < expected:
        raise RuntimeError("unreviewed rank geometry")
    if not 0 <= local_rank < expected or torch.cuda.device_count() != expected:
        raise RuntimeError("each rank must see all allocated GPUs on this node")
    total_start = time.perf_counter()
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none",
                                 "-k", "on", "g", str(expected)])
    try:
        if (lmp.extract_setting("world_size") != expected or
                lmp.extract_setting("world_rank") != rank):
            raise RuntimeError("LAMMPS and Slurm MPI identity disagree")
        if not lmp.has_package("KOKKOS") or not lmp.has_style("pair", "mliap/kk"):
            raise RuntimeError("MPI LAMMPS needs KOKKOS and mliap/kk")
        activate_mliappy_kokkos(lmp)
        for command in (
            # Site MPICH/CXI with peer-visible allocated GPUs passed the
            # distinct 2- and 4-rank functional gates; this is not a
            # two-node or throughput qualification.
            "package kokkos newton on neigh half gpu/aware on",
            "units metal", "atom_style atomic/kk", "newton on", "boundary p p p",
            "lattice diamond 5.43",
            f"region box block 0 {args.cells} 0 {args.cells} 0 {args.cells} units lattice",
            "create_box 1 box", "create_atoms 1 box",
            "run_style verlet/kk", "mass 1 28.085",
            "atom_modify sort 0 0.0",
            f"pair_style mliap/kk unified {args.model} 0",
            "pair_coeff * * Si",
            "velocity all create 300 20260924 mom yes rot yes dist gaussian",
            "timestep 0.0001",  # metal units: 0.1 fs
            "fix integrate all nve/kk",
            "fix bath all langevin 300 300 0.1 20260925",
            "thermo_style custom step temp pe",
        ):
            lmp.command(command)
        atoms = 8 * args.cells**3
        if lmp.get_natoms() != atoms:
            raise RuntimeError("LAMMPS atom count differs from silicon supercell")
        started = time.perf_counter()
        lmp.command("run 0")
        lmp.command(f"run {args.steps}")
        elapsed = time.perf_counter() - started
        energy = float(lmp.get_thermo("pe"))
        if not math.isfinite(energy) or elapsed <= 0:
            raise RuntimeError("non-finite LAMMPS result")
        memory = [torch.cuda.memory_allocated(index) for index in range(expected)]
        selected = max(range(expected), key=memory.__getitem__)
        if memory[selected] <= 0 or selected != local_rank:
            raise RuntimeError("MPI rank did not select its distinct local GPU")
    finally:
        lmp.close()

    record = {
        "format": 1, "state": "completed", "benchmark": "single-trajectory-domain",
        "engine": "lammps-mliap-kokkos-mpi", "model_sha256": MODEL_SHA256,
        "cells": args.cells, "atoms": atoms, "trajectories": 1,
        "nodes": nodes, "gpus": expected, "rank": rank, "local_rank": local_rank,
        "visible_gpus": expected, "selected_device_index": selected,
        "torch_memory_bytes": memory,
        "temperature_k": 300.0, "timestep_fs": 0.1,
        "warmup_steps": 0, "measured_steps": args.steps,
        "measured_seconds": elapsed, "total_seconds": time.perf_counter() - total_start,
        "atom_steps_per_second": atoms * args.steps / elapsed,
        "final_energy_ev": energy,
    }
    path = args.output_dir / f"rank-{rank}.json"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    run(arguments())
