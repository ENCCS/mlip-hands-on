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

Each row completes eight simulations, each with 10 warmup and 200 measured
steps at 0.1 fs. `processes` is the number of LAMMPS client processes; the
ALCHEMI batch is one process with eight replicas. The two engines used NVE
velocity-Verlet, but initial velocities were independent, not atom-by-atom
matched. **Time to finish all eight** is the measured group wall time: it
includes client startup, model loading, warmup, and MD; it
excludes queue wait, native archive extraction, and MPS controller startup.
The secondary aggregate rate counts completed MD steps across all eight
simulations per second: `8 × 200 / group wall seconds`. It is not the speed
of one trajectory or a standard MD performance unit.

```{code-cell} ipython3
import csv
from pathlib import Path
from IPython.display import Markdown, display

episode_dir = Path.cwd() / "episodes" if (Path.cwd() / "episodes").is_dir() else Path.cwd()
with (episode_dir / "reviewed-shared-gpu-nve.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

assert len(rows) == 16 and {int(row["atoms_per_replica"]) for row in rows} == {64, 512}
display(Markdown(
    "| Atoms/simulation | Method | Clients | Time to finish all 8 (s) | MD steps across all 8/s |\n"
    "| ---: | --- | ---: | ---: | ---: |\n" +
    "\n".join(
        f"| {r['atoms_per_replica']} | {r['method']} | {r['processes']} | "
        f"{r['group_wall_seconds']} | {r['replica_steps_per_second']} |"
        for r in rows
    )
))
```

## Generate a comparison figure

The chart shows the time to finish all eight simulations; **shorter is
better** for this fixed amount of work. The table above retains every tested
client count. To keep the comparison readable, the chart shows ALCHEMI and
the two eight-process LAMMPS modes, with ordinary sharing and CUDA MPS kept
separate. These are complete-workflow timings, not force-kernel speedups.

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt

selected = {("ALCHEMI batch", "1"), ("LAMMPS ordinary sharing", "8"),
            ("LAMMPS CUDA MPS", "8")}
labels = ["ALCHEMI batch", "LAMMPS ordinary sharing", "LAMMPS CUDA MPS"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, atoms in zip(axes, (64, 512)):
    subset = [r for r in rows if int(r["atoms_per_replica"]) == atoms
              and (r["method"], r["processes"]) in selected]
    by_method = {r["method"]: float(r["group_wall_seconds"]) for r in subset}
    ax.barh(labels, [by_method[label] for label in labels])
    ax.set_title(f"Eight simulations × {atoms} atoms")
    ax.set_xlabel("Time to finish all eight (s; shorter is better)")
    ax.invert_yaxis()
fig.suptitle("One GH200, completed NVE workflows (one run per case)")
fig.tight_layout()
plt.show()
```

## Interpret the measurement

At 64 atoms per simulation, ALCHEMI finished all eight in 18.745 s;
the fastest tested LAMMPS mode took 88.336 s, about 4.71 times as long.
At 512 atoms, the corresponding times were 26.599 s and 90.782 s,
about 3.41 times as long. This is not a repeated-run
estimate, intrinsic kernel speedup, trajectory-equivalence result, or GPU
memory limit. The 4,096-atom matrix timed out before its eighth row; do not
fill that gap by extrapolation.

For a new comparison, record model and software hashes, GPU occupancy,
initial positions and velocities, and the exact warmup and timed-step counts.
Run each case repeatedly without another workload on the GPU; report median
and range for time to finish all eight. To compare scientific
results, inspect energies and forces at matched states separately from timing.
At 64 atoms, setup and model loading can occupy more of the total time than
at a much larger atom count.
