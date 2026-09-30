# Paste this whole file into one notebook code cell on an allocated NVIDIA GPU.
# Use only a trusted original MACE checkpoint; torch.load deserializes it.
import math
import os

import torch
from ase.build import bulk
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.dynamics.hooks import LoggingHook
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

model_path = os.environ.get("MLIP_MODEL_PATH", "/models/mace.model")
replicas = int(os.environ.get("MLIP_REPLICAS", "1"))
steps = int(os.environ.get("MLIP_STEPS", "200"))
log_every = 1000
device = torch.device("cuda:0")

checkpoint = torch.load(model_path, weights_only=False, map_location=device)
model = MACEWrapper(checkpoint.to(device=device, dtype=torch.float32)).eval()
atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

systems = []
for index in range(replicas):
    system = AtomicData.from_atoms(atoms, device=device, dtype=torch.float32)
    system.add_node_property("forces", torch.zeros((len(atoms), 3), device=device))
    system.add_system_property("energy", torch.zeros((1, 1), device=device))
    rng = torch.Generator(device=device).manual_seed(20260924 + index)
    velocity = torch.randn((len(atoms), 3), device=device, generator=rng)
    velocity *= math.sqrt(KB_EV * 300.0 / 28.085)
    velocity -= velocity.mean(dim=0, keepdim=True)
    system.add_node_property("velocities", velocity)
    systems.append(system)
batch = Batch.from_data_list(systems, device=device)
neighbors = NeighborListHook(model.model_config.neighbor_config,
                             stage=DynamicsStage.BEFORE_COMPUTE)
neighbors._rebuild(batch)
initial = model(batch)
batch.forces.copy_(initial["forces"])
batch.energy.copy_(initial["energy"])

# from nvalchemi.dynamics import NVTNoseHoover
# md = NVTNoseHoover(model=model, dt=0.1, temperature=300.0,
#                    thermostat_time=10.0, n_steps=steps)
md = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.5,
                 n_steps=steps, random_seed=20260925)
md.register_hook(neighbors)

def print_progress(step, rows):
    # One row per independent trajectory; report their means, not their sum.
    energy = sum(row["energy"] for row in rows) / len(rows)
    temperature = sum(row["temperature"] for row in rows) / len(rows)
    print(f"step {step:5d} | mean potential energy {energy:10.4f} eV "
          f"| mean temperature {temperature:7.1f} K", flush=True)

with LoggingHook(backend="custom", frequency=log_every,
                 writer_fn=print_progress) as progress:
    md.register_hook(progress)
    batch = md.run(batch)

print(f"Completed {replicas} trajectories of {len(atoms)} Si atoms for {steps} steps")
print("Final potential energies (eV):", batch.energy.reshape(-1).detach().cpu().tolist())
