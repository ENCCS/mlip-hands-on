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

# One silicon simulation on more than one GPU

How many GPUs should you request for one large molecular-dynamics (MD)
simulation? The useful questions are whether the system fits, how long the
run takes, and whether adding GPUs saves enough time to justify using them.
This page uses measured LAMMPS/MACE results from one GH200 node on
Arrhenius or JUPITER.
It is different from [running eight independent simulations on one GPU](05-batched-md.md).

## Start with one GPU

The size of your system changes the time needed for each MD step. Separate
one-GH200 LAMMPS/MACE size sweeps gave these preliminary results after ten
warmup steps. Each row is **one** run of 100 measured steps, not a median.
The tabs select displayed site evidence; they do not change a notebook kernel.

::::{tab-set}
:sync-group: site

:::{tab-item} Arrhenius
:sync: arrhenius
| Silicon atoms | Time for 100 measured MD steps |
| ---: | ---: |
| 512 | 35.4 s |
| 8,000 | 43.7 s |
| 32,768 | 117.1 s |
:::

:::{tab-item} JUPITER
:sync: jupiter
| Silicon atoms | Time for 100 measured MD steps |
| ---: | ---: |
| 512 | 30.6 s |
| 8,000 | 43.4 s |
| 32,768 | 120.2 s |

JUPITER allocated a full four-GH200 Booster node for this diagnostic,
although this sweep used one GPU. These are not controlled cross-site
comparisons of the machines.
:::

::::

In separate short capacity probes with the pinned model, one GH200 completed
a 39,304-atom silicon run on **each site**. The next tested cubic size,
46,656 atoms, failed with a CUDA out-of-memory error on each site. The
JUPITER failure occurred during the two-step warmup; it was not a completed
MD run. Both successful capacity probes used only ten measured MD steps.
This brackets **the tested workload on each site**, not the maximum size for
every model or trajectory. The one-GPU size and capacity probes used a
separate native runner from the MPI scaling jobs below; do not combine their
times into one speedup calculation. None of these probes establishes the
largest system four GPUs can run together.

## Does adding GPUs finish the same system sooner?

**Strong scaling** keeps the number of atoms fixed and gives the *same
coupled system* more GPUs. Here LAMMPS divided one 32,768-atom silicon system
among one, two, or four GH200s in the same node. Each configuration ran 100
cold MD steps at a 0.1-fs timestep with the same MACE model. Each site table
gives the median of three completed repetitions per GPU count. Both site
series keep the 32,768-atom coupled system fixed.

::::{tab-set}
:sync-group: site

:::{tab-item} Arrhenius
:sync: arrhenius
| GH200 GPUs | Median MD time for 100 steps | Speedup over one GPU |
| ---: | ---: | ---: |
| 1 | 119.8 s | 1.00× |
| 2 | 78.4 s | 1.53× |
| 4 | 50.8 s | 2.36× |
:::

:::{tab-item} JUPITER
:sync: jupiter
| GH200 GPUs | Median MD time for 100 steps | Speedup over one GPU |
| ---: | ---: | ---: |
| 1 | 122.6 s | 1.00× |
| 2 | 73.4 s | 1.67× |
| 4 | 47.7 s | 2.57× |

All three JUPITER GPU counts allocated a full Booster node. Each MPI rank
observed its own single visible GPU; the measured duration is the slowest
rank's duration for that repetition.
:::

::::

Four GPUs finished this short workload sooner on both sites, but not four
times sooner. On Arrhenius, four GPUs for 50.8 seconds occupy about 203
GPU-seconds, compared with about 120 GPU-seconds on one GPU. Choose according to whether
shorter waiting time or lower GPU use matters more for your research. These
numbers are application timings, not a prediction of scheduler charges.

The Arrhenius measured interval includes LAMMPS `run 0` and the 100 steps,
but excludes
queue wait, process startup, and extraction of the native runtime archive.
The runs use different rank counts and Langevin random streams; their timing
results do **not** establish identical trajectories or scientific agreement.
Three short samples on each site show limited run-to-run variation, not the
sustained rate of a long production trajectory. These are separate
measurements with different runtime stacks; do not infer an Arrhenius-versus-
JUPITER hardware ranking from the nearby numbers.

The MPI-enabled example below is the reviewed *job template* for this
32,768-atom case. Check its account, partition, time, memory, and GPU request
against the current site before any manual submission. Change the MPI rank,
GPU, CPU, and memory requests together for a separately reviewed two- or
four-GPU job. Opening this page or running a notebook cell submits nothing.

:::{warning}
Do not treat `sbatch --test-only` as proof that a reservation or allocation
will actually run. Verify current site policy and the real job outcome.
:::

```{literalinclude} ../scripts/lammps-mpi-size-benchmark.sbatch
:language: bash
:lines: 1-16
:lineno-match:
```

Read the [complete job script](../scripts/lammps-mpi-size-benchmark.sbatch)
before adapting or submitting it; the excerpt hides setup, checks, and the
actual launch command.

## What about weak scaling and larger systems?

**Weak scaling** would increase the number of atoms as GPUs are added, so
each GPU handles roughly the same amount of work. We have **not** measured
a comparable one-/two-/four-GPU weak-scaling series for this LAMMPS/MACE
system. Do not read the strong-scaling table as a weak-scaling result.

We also have not measured the four-GPU atom limit. To find a useful bound,
predeclare increasing system sizes, keep the model and run settings fixed,
record both successful jobs and explicit memory failures, and stop at the
first resource limit. Four GPUs do not act as one pooled GPU-memory device;
domain decomposition introduces per-rank data and communication. The largest
completed case is a **tested workload**, not an intrinsic atom limit. See
[Limits and provenance](../reference/limits.md).

ALCHEMI Toolkit has a separate `DomainParallel` route for dividing one
system across GPUs. Its current multi-GPU records include an unresolved
rank-count energy difference, so this page does not use those numbers for
a cross-engine scaling claim.

JUPITER has a separate [per-rank GPU binding and native MPI setup](../setup/jupiter.md).
An eight-GPU two-node run passed a short functional check there. Ten measured
steps on 512 atoms are not a multi-node scaling study; do not plot those
elapsed times as speedups.

For many small *independent* runs instead of one coupled system, use the
[batched-MD chapter](05-batched-md.md) and the
[reviewed eight-simulation comparison](07-reviewed-results.md). Its measure
is the time to finish the whole set, not strong or weak scaling of one
simulation.
