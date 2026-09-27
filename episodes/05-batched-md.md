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

# Batch independent trajectories on one GPU

Eight copies of the same 64-atom cell receive different seeded velocities.
They remain **eight independent simulations**, not one coupled 512-atom
cell. `Batch.from_data_list` gathers their atomic data so one ALCHEMI run
can advance all replicas together.

The first cell measures one trajectory. The second measures eight. They run
sequentially on the same allocated GPU; do not execute another GPU notebook
at the same time.

```{code-cell} ipython3
import json
import subprocess
from pathlib import Path
from IPython.display import Markdown, display

lesson = Path.cwd().parent if Path.cwd().name == "episodes" else Path.cwd()
single_call = subprocess.run(
    ["bash", str(lesson / "scripts/run-alchemi.sh"), "--replicas", "1", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
single = json.loads(single_call.stdout.strip().splitlines()[-1])
display(Markdown(
    f"One trajectory: **{single['measured_seconds']:.3f} s** for "
    f"{single['measured_steps']} timed steps "
    f"({single['replica_steps_per_second']:.2f} replica-steps/s)."
))
```

```{code-cell} ipython3
batch_call = subprocess.run(
    ["bash", str(lesson / "scripts/run-alchemi.sh"), "--replicas", "8", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
batch = json.loads(batch_call.stdout.strip().splitlines()[-1])
ratio = batch["replica_steps_per_second"] / single["replica_steps_per_second"]
display(Markdown(
    "| Replicas | Atoms each | MD time (s) | Aggregate replica-steps/s |\n"
    "| ---: | ---: | ---: | ---: |\n"
    f"| 1 | {single['atoms_per_replica']} | {single['measured_seconds']:.3f} | "
    f"{single['replica_steps_per_second']:.2f} |\n"
    f"| 8 | {batch['atoms_per_replica']} | {batch['measured_seconds']:.3f} | "
    f"{batch['replica_steps_per_second']:.2f} |\n\n"
    f"The eight-replica aggregate rate is **{ratio:.2f}×** the single-replica rate."
))
```

The numerator counts all completed replica-steps. The ratio does **not**
mean that any individual trajectory ran that much faster. These two short
runs are an illustration, not the reviewed repeated benchmark. Larger
systems may already fill the GPU, leaving less benefit from batching.

:::{note}
To run NVT instead, change `--integrator nve` to `langevin` in both cells.
The ALCHEMI code uses `NVTLangevin`; native LAMMPS can use `langevin` with
`nve/kk` or a Nose–Hoover `nvt/kk` fix. The controls and discretizations
are not interchangeable merely because they share a target temperature.
Keep NVE and NVT results in separate tables.
:::
