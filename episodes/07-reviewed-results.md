---
mystnb:
  execution_mode: force
  render_markdown_format: gfm
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

# Read a completed shared-GPU benchmark

Use the checked-in measurements to compare ALCHEMI batching with eight native
LAMMPS processes sharing one GH200. Each run advances eight **independent**
silicon/MACE trajectories. This offline page reads a small CSV; its cells do **not** run MD,
request an allocation, or contact Slurm.

The CSV contains two completed NVE matrices, at 64 and 512 atoms per
replica. An incomplete 4,096-atom matrix is excluded. These results are
independent site measurements, not the timings from your notebook session.

## Reference measurements

The read-only cell below renders the reviewed CSV as a table in both the
published handout and the live notebook. Re-running it does not run MD or
change the reference data.

Each row completes eight replicas, each with 10 warmup and 200 measured
steps at 0.1 fs. `processes` is the number of LAMMPS client processes; the
ALCHEMI batch is one process with eight replicas. The two engines used NVE
velocity-Verlet, but initial velocities were independent, not atom-by-atom
matched. Group wall includes startup, model loading, warmup, and MD; it
excludes queue wait, native archive extraction, and MPS controller startup.

```{code-cell} ipython3
import csv
from pathlib import Path
from IPython.display import Markdown, display

episode_dir = Path.cwd() / "episodes" if (Path.cwd() / "episodes").is_dir() else Path.cwd()
with (episode_dir / "reviewed-shared-gpu-nve.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

assert len(rows) == 16 and {int(row["atoms_per_replica"]) for row in rows} == {64, 512}
display(Markdown(
    "| Atoms/replica | Method | Clients | Group wall (s) | Replica-steps/s |\n"
    "| ---: | --- | ---: | ---: | ---: |\n" +
    "\n".join(
        f"| {r['atoms_per_replica']} | {r['method']} | {r['processes']} | "
        f"{r['group_wall_seconds']} | {r['replica_steps_per_second']} |"
        for r in rows
    )
))
```

## Generate a comparison figure

The bars below are **complete-workflow aggregate throughput**, not force
kernel speedups. Keep ordinary GPU sharing and CUDA MPS separate.

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt

labels = [f"{r['method']} ({r['processes']})" for r in rows[:8]]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, atoms in zip(axes, (64, 512)):
    subset = [r for r in rows if int(r["atoms_per_replica"]) == atoms]
    ax.barh(labels, [float(r["replica_steps_per_second"]) for r in subset])
    ax.set_title(f"Eight replicas × {atoms} atoms")
    ax.set_xlabel("Aggregate replica-steps/s")
    ax.invert_yaxis()
fig.suptitle("One GH200, completed NVE workflows (one run per size)")
fig.tight_layout()
plt.show()
```

## Interpret the measurement

At 64 atoms/replica, ALCHEMI's 85.358 replica-steps/s versus the fastest
LAMMPS row's 18.113 is about 4.71× for this one complete workflow. At 512
atoms/replica, 60.153 versus 17.625 is about 3.41×. This is not a repeated-run
estimate, intrinsic kernel speedup, trajectory-equivalence result, or GPU
memory limit. The 4,096-atom matrix timed out before its eighth row; do not
fill that gap by extrapolation.

For a new comparison, record model and software hashes, GPU occupancy,
initial positions and velocities, and the exact warmup and timed-step counts.
Run each case repeatedly without another workload on the GPU; report median
and range for group wall time and aggregate throughput. To compare scientific
results, inspect energies and forces at matched states separately from timing.
At 64 atoms, setup and model loading can occupy more of the total time than
at a much larger atom count.
