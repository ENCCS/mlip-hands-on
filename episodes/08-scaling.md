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
This page uses measured LAMMPS/MACE results from one Arrhenius GH200 node.
It is different from [running eight independent simulations on one GPU](05-batched-md.md).

## Start with one GPU

The size of your system changes the time needed for each MD step. A separate
one-GH200 LAMMPS/MACE size sweep gave these preliminary results after ten
warmup steps. Each row is **one** run of 100 measured steps, not a median:

| Silicon atoms | Approximate time for 100 measured MD steps |
| ---: | ---: |
| 512 | 35.4 s |
| 8,000 | 43.7 s |
| 32,768 | 117.1 s |

In a separate short capacity probe with the pinned model, one GH200
completed a 39,304-atom silicon run; the next tested cubic size, 46,656
atoms, failed with a CUDA out-of-memory error. Both capacity probes used
only ten measured MD steps. This brackets **that tested workload**, not
the maximum size for every model or trajectory. The one-GPU size and
capacity probes used a separate native runner from the MPI scaling jobs
below; do not combine their times into one speedup calculation. None of
these probes establishes the largest system four GPUs can run together.

## Does adding GPUs finish the same system sooner?

**Strong scaling** keeps the number of atoms fixed and gives the *same
coupled system* more GPUs. Here LAMMPS divided one 32,768-atom silicon system
among one, two, or four GH200s in the same node. Each configuration ran 100
cold MD steps at a 0.1-fs timestep with the same MACE model. The table gives
the median of three distinct completed jobs per GPU count.

| GH200 GPUs | Median MD time for 100 steps | Speedup over one GPU |
| ---: | ---: | ---: |
| 1 | 119.8 s | 1.00× |
| 2 | 78.4 s | 1.53× |
| 4 | 50.8 s | 2.36× |

Four GPUs finished this short workload sooner, but not four times sooner.
Using four GPUs for 50.8 seconds also occupies about 203 GPU-seconds,
compared with about 120 GPU-seconds on one GPU. Choose according to whether
shorter waiting time or lower GPU use matters more for your research. These
numbers are application timings, not a prediction of scheduler charges.

The measured interval includes LAMMPS `run 0` and the 100 steps, but excludes
queue wait, process startup, and extraction of the native runtime archive.
The runs use different rank counts and Langevin random streams; their timing
results do **not** establish identical trajectories or scientific agreement.
Three short samples show limited run-to-run variation, not the sustained
rate of a long production trajectory.

The MPI-enabled example below is the reviewed *job template* for this
32,768-atom case. Check its account, partition, time, memory, and GPU request
against the current site before any manual submission. Change the MPI rank,
GPU, CPU, and memory requests together for a separately reviewed two- or
four-GPU job. Opening this page or running a notebook cell submits nothing;
do not treat `sbatch --test-only` as proof that a reservation or allocation
will actually run.

```{literalinclude} ../scripts/lammps-mpi-size-benchmark.sbatch
:language: bash
:lines: 1-16
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
One-, two-, and four-GPU one-node runs and an eight-GPU two-node run have
passed short functional checks there. Ten measured steps on 512 atoms are
not a scaling study; do not plot those elapsed times as speedups.

For many small *independent* runs instead of one coupled system, use the
[batched-MD chapter](05-batched-md.md) and the
[reviewed eight-simulation comparison](07-reviewed-results.md). Its measure
is the time to finish the whole set, not strong or weak scaling of one
simulation.
