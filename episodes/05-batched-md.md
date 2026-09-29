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

From the lesson checkout, the equivalent terminal commands are:

```bash
bash scripts/run-alchemi.sh --replicas 1 --cells 2 --integrator nve --warmup 10 --steps 200
bash scripts/run-alchemi.sh --replicas 8 --cells 2 --integrator nve --warmup 10 --steps 200
```

The notebook times each complete command, including setup and model loading.
The runner also reports MD-only time; those are different timing windows.

```{code-cell} ipython3
:tags: [hide-input]
import json
import subprocess
from time import perf_counter
from pathlib import Path
from IPython.display import Markdown, display

lesson = Path.cwd().parent if Path.cwd().name == "episodes" else Path.cwd()
single_start = perf_counter()
single_call = subprocess.run(
    ["bash", str(lesson / "scripts/run-alchemi.sh"), "--replicas", "1", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
single_wall = perf_counter() - single_start
single = json.loads(single_call.stdout.strip().splitlines()[-1])
display(Markdown(
    f"Time to finish one simulation: **{single_wall:.3f} s** "
    f"({single['measured_steps']} measured MD steps; "
    f"MD-only time {single['measured_seconds']:.3f} s)."
))
```

```{code-cell} ipython3
:tags: [hide-input]
batch_start = perf_counter()
batch_call = subprocess.run(
    ["bash", str(lesson / "scripts/run-alchemi.sh"), "--replicas", "8", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
batch_wall = perf_counter() - batch_start
batch = json.loads(batch_call.stdout.strip().splitlines()[-1])
display(Markdown(
    "| Simulations | Atoms each | Time to finish all (s) | MD-only time (s) |\n"
    "| ---: | ---: | ---: | ---: |\n"
    f"| 1 | {single['atoms_per_replica']} | {single_wall:.3f} | "
    f"{single['measured_seconds']:.3f} |\n"
    f"| 8 | {batch['atoms_per_replica']} | {batch_wall:.3f} | "
    f"{batch['measured_seconds']:.3f} |"
))
```

The two rows do different amounts of work, so their wall times alone are not
a speedup ratio. To compare methods completing the *same eight simulations*,
use the [reviewed benchmark](07-reviewed-results.md). If an aggregate rate is
useful, calculate it as `simulations × measured steps / time to finish all`
(completed MD steps across all simulations per second); it does not describe
the speed of any single trajectory. These short live runs are illustrations,
not the reviewed benchmark. Larger systems may already fill the GPU, leaving
less benefit from batching.

:::{note}
To run NVT instead, change `--integrator nve` to `langevin` in both cells.
The ALCHEMI code uses `NVTLangevin`; native LAMMPS can use `langevin` with
`nve/kk` or a Nose–Hoover `nvt/kk` fix. The controls and discretizations
are not interchangeable merely because they share a target temperature.
Keep NVE and NVT results in separate tables.
:::
