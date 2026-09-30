---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Run one silicon trajectory

Start with 64 silicon atoms on one allocated GPU. The ALCHEMI program
[`alchemi_si_one_cell.py`](../../examples/alchemi_si_one_cell.py) creates
the diamond cell, loads the original MACE checkpoint, and runs Langevin MD.
The LAMMPS [input file](../../examples/lammps_mace.in) uses the exported
model with NVE integration and a Langevin thermostat. Both use a 0.1 fs
timestep, but their thermostat implementations need not yield identical
trajectories.

Load the notebook's ALCHEMI adapter. It sends the *visible Python cell* to
the selected SIF on the allocated GPU; the documentation kernel does not
need to import ALCHEMI itself.

```{code-cell} ipython3
%run ../../scripts/notebook_alchemi.py
```

The following cell is the actual one-trajectory simulation. The same source
is in [`alchemi_si_one_cell.py`](../../examples/alchemi_si_one_cell.py), so it
can also be run from a shell without a notebook.

```{code-cell} ipython3
%%alchemi 1 200
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
```

It prints a completion line and final potential energy. Increase the step
count to 2,000 to see its 1,000-step energy/temperature progress lines.

The LAMMPS calculation uses its CLI and the input shown below. The input
creates the silicon cell, selects the exported MACE potential, and sets the
time step and thermostat.

```{literalinclude} ../../examples/lammps_mace.in
:language: text
:linenos:
```

Here `-k on g 1` selects one Kokkos GPU, `-sf kk` selects Kokkos styles,
`-in` names the input file, and `-var` supplies values used in that file.
On Arrhenius, the notebook starts native MPICH LAMMPS in its own
`srun --mpi=pmi2` step. JUPITER's native OpenMPI build runs directly in
the GPU-bound notebook step.

```{code-cell} ipython3
%%bash
cd ../..
source "scripts/${MLIP_SITE}-lammps-env.sh"
launch=()
if [[ "$MLIP_SITE" == arrhenius ]]; then
  launch=(srun --mpi=pmi2 --nodes=1 --ntasks=1 --gpus=1)
fi
"${launch[@]}" "$MLIP_LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -log none -in examples/lammps_mace.in \
  -var model "$MLIP_MLIAP_MODEL" -var cells 2 -var warmup 10 \
  -var steps 200 -var seed 20260924 -var bath_seed 20260925
```

LAMMPS prints step, atom count, temperature, and potential energy. The
input currently prints every ten steps; it includes a commented 1,000-step
option for longer runs. The next episode repeats independent ALCHEMI
trajectories on one GPU.
