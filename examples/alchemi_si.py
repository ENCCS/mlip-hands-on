#!/usr/bin/env python3
"""Time independent 64-atom Si/MACE trajectories in one ALCHEMI batch."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from ase.build import bulk
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE, NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


MODEL_SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_batch(device, replicas, cells=2):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((cells, cells, cells))
    count = len(atoms)
    systems = []
    for index in range(replicas):
        item = AtomicData.from_atoms(atoms, device=device, dtype=torch.float32)
        item.add_node_property("forces", torch.zeros((count, 3), device=device))
        item.add_system_property("energy", torch.zeros((1, 1), device=device))
        generator = torch.Generator(device=device).manual_seed(20260924 + index)
        velocity_scale = math.sqrt(KB_EV * 300.0 / 28.085)
        velocities = torch.randn((count, 3), device=device, generator=generator) * velocity_scale
        velocities -= velocities.mean(dim=0, keepdim=True)
        item.add_node_property("velocities", velocities)
        systems.append(item)
    return Batch.from_data_list(systems, device=device)


def run_steps(model, batch, steps, seed, integrator="langevin"):
    if integrator == "nve":
        dynamics = NVE(model=model, dt=0.1, n_steps=steps)
    else:
        dynamics = NVTLangevin(
            model=model, dt=0.1, temperature=300.0, friction=0.5,
            n_steps=steps, random_seed=seed,
        )
    dynamics.register_hook(NeighborListHook(
        model.model_config.neighbor_config,
        stage=DynamicsStage.BEFORE_COMPUTE,
    ))
    result = dynamics.run(batch)
    if dynamics.step_count != steps:
        raise RuntimeError("ALCHEMI did not finish the requested steps")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--integrator", choices=("langevin", "nve"), default="langevin")
    parser.add_argument("--cells", type=int, choices=(2, 4, 8, 10), default=2)
    args = parser.parse_args()
    if not args.model.is_file() or sha256(args.model) != MODEL_SHA256:
        parser.error("original MACE checkpoint identity does not match")
    if not 1 <= args.replicas <= 64 or not 1 <= args.steps <= 10000:
        parser.error("replicas must be 1..64 and steps 1..10000")
    atoms_per_replica = 8 * args.cells**3
    if atoms_per_replica * args.replicas > 64000:
        parser.error("unreviewed total atom count")
    if not 0 <= args.warmup <= 1000:
        parser.error("warmup must be 0..1000")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("exactly one visible CUDA GPU is required")

    total_started = time.perf_counter()
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.model, weights_only=False, map_location=device)
    model = MACEWrapper(checkpoint.to(device=device, dtype=torch.float32)).eval()
    batch = make_batch(device, args.replicas, args.cells)
    if args.integrator == "nve":
        # NVE's first half-kick needs the force at the initial positions.
        hook = NeighborListHook(model.model_config.neighbor_config,
                                stage=DynamicsStage.BEFORE_COMPUTE)
        hook._rebuild(batch)
        initial = model(batch)
        batch.forces.copy_(initial["forces"])
        batch.energy.copy_(initial["energy"])
    setup_started = time.perf_counter()
    if args.warmup:
        batch = run_steps(model, batch, args.warmup, 20260924, args.integrator)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    batch = run_steps(model, batch, args.steps, 20260925, args.integrator)
    torch.cuda.synchronize()
    measured_seconds = time.perf_counter() - started
    total_seconds = time.perf_counter() - total_started
    energies = batch.energy.reshape(-1)
    if energies.numel() != args.replicas or not torch.isfinite(energies).all():
        raise RuntimeError("non-finite or missing replica energy")
    print(json.dumps({
        "format": 1, "engine": "alchemi", "state": "completed",
        "integrator": args.integrator,
        "replicas": args.replicas, "atoms_per_replica": atoms_per_replica,
        "warmup_steps": args.warmup, "measured_steps": args.steps,
        "timestep_fs": 0.1, "temperature_k": 300.0,
        "warmup_seconds": setup_seconds, "measured_seconds": measured_seconds,
        "total_seconds": total_seconds,
        "replica_steps_per_second": args.replicas * args.steps / measured_seconds,
        "end_to_end_replica_steps_per_second": args.replicas * args.steps / total_seconds,
        "mean_replica_latency_seconds": measured_seconds,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "model_sha256": MODEL_SHA256,
        "local_rank": int(os.environ.get("MLIP_LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
