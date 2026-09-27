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

# Run the same silicon case with LAMMPS

LAMMPS loads the validated ML-IAP export of the MACE checkpoint. ML-IAP
connects the potential to LAMMPS, while `/kk` styles use Kokkos for the
supported GPU work. The relevant input is visible in `examples/lammps_si.py`:

```{literalinclude} ../examples/lammps_si.py
:language: python
:start-at: for command in (
:end-at: lmp.command("run 0")
```

With the MPI-enabled native runtime extracted and selected as
`MLIP_NATIVE_PREFIX`, the first cell runs one 64-atom NVE trajectory on one
GPU. It is a one-rank run of an MPI-capable binary, not an MPI scaling test.

```{code-cell} ipython3
import json
import subprocess
from pathlib import Path
from IPython.display import Markdown, display

lesson = Path.cwd().parent if Path.cwd().name == "episodes" else Path.cwd()
completed = subprocess.run(
    ["bash", str(lesson / "scripts/run-lammps.sh"), "--replicas", "1", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
lammps_one = json.loads(completed.stdout.strip().splitlines()[-1])
display(Markdown(
    f"One LAMMPS trajectory: **{lammps_one['measured_seconds']:.3f} s** "
    f"for {lammps_one['measured_steps']} timed steps."
))
```

For eight independent simulations on one GPU, there are three distinct
workloads:

1. Run eight LAMMPS trajectories **sequentially**. This is a useful simple
   baseline, but it does not share the GPU concurrently.
2. Run eight independent LAMMPS processes **at once** on the same GPU.
   Ordinary CUDA context sharing lets them take turns using GPU resources.
3. Start a job-local CUDA MPS controller, then run the eight processes.
   MPS can reduce some context-scheduling overhead but is not a guarantee of
   speedup. It must be confined to the allocation and stopped afterwards.

ALCHEMI's single-process `Batch` is none of these LAMMPS modes. Compare
the completed **group wall time** and aggregate replica-step rate for the
same number of replicas, atoms, steps, model, and GPU. The
[reviewed-results episode](07-reviewed-results.md) keeps ordinary sharing
and MPS in separate rows.

The supplied group runner starts eight independent one-replica clients.
`sequential` waits for each client before starting the next; `plain` uses
ordinary GPU sharing. `mps` starts a controller with private,
job-local pipe and log directories and stops it on exit. These are optional
longer cells; run either once in a suitably long allocation, not alongside
another GPU exercise.

```{code-cell} ipython3
sequential_call = subprocess.run(
    ["bash", str(lesson / "scripts/run-lammps-group.sh"), "sequential", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
sequential = json.loads(sequential_call.stdout.strip().splitlines()[-1])
display(Markdown(
    f"Eight sequential clients: **{sequential['group_wall_seconds']:.2f} s** "
    f"group wall, **{sequential['replica_steps_per_second']:.2f}** replica-steps/s."
))
```

```{code-cell} ipython3
plain_call = subprocess.run(
    ["bash", str(lesson / "scripts/run-lammps-group.sh"), "plain", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
plain = json.loads(plain_call.stdout.strip().splitlines()[-1])
display(Markdown(
    f"Eight ordinary-sharing clients: **{plain['group_wall_seconds']:.2f} s** "
    f"group wall, **{plain['replica_steps_per_second']:.2f}** replica-steps/s."
))
```

```{code-cell} ipython3
mps_call = subprocess.run(
    ["bash", str(lesson / "scripts/run-lammps-group.sh"), "mps", "--cells", "2",
     "--integrator", "nve", "--warmup", "10", "--steps", "200"],
    check=True, capture_output=True, text=True,
)
mps = json.loads(mps_call.stdout.strip().splitlines()[-1])
display(Markdown(
    f"Eight MPS clients: **{mps['group_wall_seconds']:.2f} s** group wall, "
    f"**{mps['replica_steps_per_second']:.2f}** replica-steps/s."
))
```
