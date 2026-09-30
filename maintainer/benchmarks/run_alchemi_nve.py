#!/usr/bin/env python3
"""Run one or more matched silicon NVE starts with pinned ALCHEMI/MACE."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from ase import Atoms
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.replicas < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("replicas and measured steps must be positive; warmup may be zero")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible NVIDIA GPU required")
    manifest = json.loads((args.starts / "manifest.json").read_text())
    if manifest["format"] != 1 or args.replicas > manifest["replicas"]:
        raise RuntimeError("starting-state manifest mismatch")
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.model, weights_only=False, map_location=device)
    model = MACEWrapper(checkpoint.to(device=device, dtype=torch.float32)).eval()
    model.model_config.active_outputs = {"energy", "forces"}
    systems = []
    for index in range(args.replicas):
        start = json.loads((args.starts / f"replica-{index}.json").read_text())
        if (start["format"] != 1 or start["replica"] != index or
                start["atoms"] != manifest["atoms_per_replica"] or
                start["velocity_unit"] != "angstrom/internal-time"):
            raise RuntimeError("starting-state record mismatch")
        atoms = Atoms("Si" * start["atoms"], positions=start["positions_a"],
                      cell=[start["cell_a"]] * 3, pbc=True)
        data = AtomicData.from_atoms(atoms, device=device, dtype=torch.float32)
        data.add_node_property("forces", torch.zeros((len(atoms), 3), device=device))
        data.add_node_property("velocities", torch.tensor(
            start["velocities_internal"], dtype=torch.float32, device=device))
        data.add_system_property("energy", torch.zeros((1, 1), device=device))
        systems.append(data)
    batch = Batch.from_data_list(systems, device=device)
    neighbors = NeighborListHook(model.model_config.neighbor_config,
                                 stage=DynamicsStage.BEFORE_COMPUTE)
    neighbors._rebuild(batch)
    initial = model(batch)
    batch.forces.copy_(initial["forces"])
    batch.energy.copy_(initial["energy"])
    if args.warmup:
        batch = NVE(model=model, dt=0.1, n_steps=args.warmup,
                    hooks=[neighbors]).run(batch)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    batch = NVE(model=model, dt=0.1, n_steps=args.steps,
                hooks=[neighbors]).run(batch)
    torch.cuda.synchronize()
    measured_seconds = time.perf_counter() - start_time
    energies = batch.energy.reshape(-1).detach().cpu().tolist()
    if len(energies) != args.replicas or not all(math.isfinite(v) for v in energies):
        raise RuntimeError("incomplete or non-finite final energy")
    print(json.dumps({"state": "completed", "engine": "alchemi", "ensemble": "NVE",
                      "replicas": args.replicas, "atoms_per_replica": manifest["atoms_per_replica"],
                      "warmup_steps": args.warmup, "measured_steps": args.steps,
                      "md_seconds": measured_seconds, "final_potential_ev": energies},
                     separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
