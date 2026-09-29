"""Relax four independent silicon structures together with ALCHEMI FIRE."""

import os
from pathlib import Path

import torch
from ase.io import read
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, ConvergenceHook
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

model_path = os.environ.get("MLIP_MODEL_PATH", "/models/mace.model")
starts_dir = Path(os.environ.get("MLIP_STARTS_DIR", "/opt/mlip/starts"))
device = torch.device("cuda:0")

checkpoint = torch.load(model_path, weights_only=False, map_location=device)
model = MACEWrapper(checkpoint.to(device=device, dtype=torch.float32)).eval()

systems = []
for path in sorted(starts_dir.glob("si-relax-*.data")):
    atoms = read(path, format="lammps-data", atom_style="atomic", units="metal")
    atoms.numbers[:] = 14  # The data file has one atom type: silicon.
    atoms.pbc = True
    system = AtomicData.from_atoms(atoms, device=device, dtype=torch.float32)
    system.add_node_property("forces", torch.zeros((len(atoms), 3), device=device))
    system.add_system_property("energy", torch.zeros((1, 1), device=device))
    system.add_node_property("velocities", torch.zeros((len(atoms), 3), device=device))
    systems.append(system)

batch = Batch.from_data_list(systems, device=device)
neighbors = NeighborListHook(model.model_config.neighbor_config,
                             stage=DynamicsStage.BEFORE_COMPUTE)
neighbors._rebuild(batch)
initial = model(batch)
batch.forces.copy_(initial["forces"])
batch.energy.copy_(initial["energy"])
initial_energy = batch.energy.reshape(-1).detach().cpu().tolist()

fire = FIRE(model=model, dt=0.5, n_steps=300,
            convergence_hook=ConvergenceHook.from_fmax(0.05))
fire.register_hook(neighbors)
batch = fire.run(batch)

force_norm = batch.forces.norm(dim=-1)
fmax = torch.zeros(batch.num_graphs, device=device)
fmax.scatter_reduce_(0, batch.batch_idx, force_norm, reduce="amax")
for name, before, after, force in zip(
    sorted(starts_dir.glob("si-relax-*.data")), initial_energy,
    batch.energy.reshape(-1).detach().cpu().tolist(), fmax.detach().cpu().tolist()
):
    print(f"{name.name}: E {before:.4f} -> {after:.4f} eV, "
          f"fmax {force:.4f} eV/A, converged {force < 0.05}")
print(f"FIRE steps: {fire.step_count}")
