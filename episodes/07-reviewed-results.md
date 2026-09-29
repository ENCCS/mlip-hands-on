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

Use completed, site-labelled measurements from **one GH200** on Arrhenius
or JUPITER to compare ALCHEMI batching with eight native LAMMPS processes
sharing that GPU. Each row advances eight **independent** silicon/MACE
trajectories. This offline page reads checked-in CSVs; its cells do **not**
run MD, request an allocation, or contact Slurm.

Each site has completed NVE matrices at 64 and 512 atoms per trajectory.
The incomplete Arrhenius 4,096-atom matrix is excluded. These are recorded
site runs, not timings from your notebook session. Do not compare the two
sites as if their runtime and startup conditions were identical.

## Reference measurements

The synchronized tabs show three easy-to-compare rows from each site's
complete matrix. They select **displayed reference data only**, not a
compute backend or a notebook kernel.

::::{tab-set}
:sync-group: site

:::{tab-item} Arrhenius
:sync: arrhenius
One Arrhenius GH200; time to finish all eight trajectories, in seconds:

| Atoms/trajectory | ALCHEMI batch | LAMMPS ordinary, 8 clients | LAMMPS MPS, 8 clients |
| ---: | ---: | ---: | ---: |
| 64 | 18.745 | 100.010 | 88.336 |
| 512 | 26.599 | 114.335 | 90.782 |

[All 16 Arrhenius rows](reviewed-shared-gpu-nve.csv).
:::

:::{tab-item} JUPITER
:sync: jupiter
One JUPITER GH200 (a full node was allocated); time to finish all eight
trajectories, in seconds:

| Atoms/trajectory | ALCHEMI batch | LAMMPS ordinary, 8 clients | LAMMPS MPS, 8 clients |
| ---: | ---: | ---: | ---: |
| 64 | 76.380 | 95.733 | 73.033 |
| 512 | 35.166 | 115.179 | 75.696 |

[All 16 JUPITER comparison rows](reviewed-shared-gpu-nve-jupiter.csv).
The private JUPITER matrix also retains a redundant ordinary-sharing
one-client row at each size. The site MPS server was observed during the
concurrent clients.
:::

::::

## Whole workflow or measured MD?

The two clocks answer different questions. **Whole workflow** is the time
from starting the ALCHEMI client process until it finishes all eight
trajectories. **Measured MD** covers only the 200 steps after ten warmup
steps, with the eight trajectories advanced together. It excludes Python
startup, model loading, initial-force preparation, and warmup.

| Site | Atoms/simulation | Whole workflow, all eight (s) | Measured MD, all eight (s) |
| --- | ---: | ---: | ---: |
| Arrhenius | 64 | 18.745 | 5.595 |
| Arrhenius | 512 | 26.599 | 13.373 |
| JUPITER | 64 | 76.380 | 4.571 |
| JUPITER | 512 | 35.166 | 13.232 |

These are **one completed run per case**, not medians. The two sites used
different runtime packaging, and JUPITER ran the 64-atom case first in its
job. Its long first whole-workflow time is therefore not evidence that its
GPU advances MD more slowly: the measured MD intervals are similar. The
records locate the difference in startup and setup, but do not identify a
single cause such as import time, filesystem caching, or GPU power policy.
For a site-performance claim, repeat cases in varied order and report
medians and ranges for *both* clocks.

The read-only cell below renders one full reviewed CSV as a table in both
the published handout and the live notebook. Change `site` in the cell to
inspect the other site; choosing a tab does not silently change a notebook
variable. Re-running the cell does not run MD or change reference data.

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
:tags: [hide-input]
import csv
from pathlib import Path
from IPython.display import Markdown, display

episode_dir = Path.cwd() / "episodes" if (Path.cwd() / "episodes").is_dir() else Path.cwd()
site = "Arrhenius"  # change to "JUPITER" to inspect its full table and figure
sources = {
    "Arrhenius": "reviewed-shared-gpu-nve.csv",
    "JUPITER": "reviewed-shared-gpu-nve-jupiter.csv",
}
with (episode_dir / sources[site]).open(newline="") as stream:
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
:tags: [hide-input]
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
fig.suptitle(f"{site} GH200: completed NVE workflows (one run per case)")
fig.tight_layout()
plt.show()
```

## Interpret the measurement

On Arrhenius, at 64 atoms per simulation, ALCHEMI finished all eight in 18.745 s;
the fastest tested LAMMPS mode took 88.336 s, about 4.71 times as long.
At 512 atoms, the corresponding times were 26.599 s and 90.782 s,
about 3.41 times as long. This is not a repeated-run
estimate, intrinsic kernel speedup, trajectory-equivalence result, or GPU
memory limit. The 4,096-atom matrix timed out before its eighth row; do not
fill that gap by extrapolation.

On JUPITER, the 64-atom ALCHEMI group took 76.380 s while its 512-atom
group took 35.166 s. The larger case is **not** intrinsically faster: the
measured MD intervals above show that startup and setup dominate the
first whole-workflow result; their precise cause remains unisolated.
Eight LAMMPS MPS clients took 73.033 s at
64 atoms and 75.696 s at 512 atoms, compared with 95.733 s and 115.179 s
under ordinary sharing. These completed cases support showing both modes,
not a general MPS or cross-site speedup claim.

For a new comparison, record model and software hashes, GPU occupancy,
initial positions and velocities, and the exact warmup and timed-step counts.
Run each case repeatedly without another workload on the GPU; report median
and range for time to finish all eight. To compare scientific
results, inspect energies and forces at matched states separately from timing.
At 64 atoms, setup and model loading can occupy more of the total time than
at a much larger atom count.
