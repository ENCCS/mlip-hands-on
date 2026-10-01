# Compare MD performance

Use these results to answer three questions: how long does one trajectory
take, how much work can one GPU complete when trajectories run together,
and does sharing a GPU between MPI ranks help a large LAMMPS trajectory?
For one trajectory across several GPUs, see [GPU scaling](08-scaling.md).

The cross-engine tables use silicon, MACE-MP-0a small, matching starting
structures and velocities, and NVE velocity-Verlet at 0.1 fs. Each run has
ten warm-up steps and 200 measured steps. Values are medians of three runs;
JUPITER tables also show the full minimum–maximum range.

**Whole-workflow time** measures launch to clean exit, including imports,
model loading, setup, warm-up, MD and shutdown. It excludes queue wait.
Compare these columns to estimate the cost of a short run. The separate
ALCHEMI MD-call and LAMMPS MD-loop timers do not have identical boundaries.
[Benchmark methods and checks](../reference/benchmarks.md) gives the details.

```{note}
The notebook demonstrations use different Langevin thermostats and initial
velocities. Their elapsed times are not the matched NVE measurements below.
```

## One trajectory: increase the atom count

Each calculation uses one GH200 GPU. Larger systems take longer even
though they use the same model and number of MD steps.

`````{tab-set}
````{tab-item} Arrhenius
:sync: arrhenius

Median seconds on one Arrhenius GH200 GPU:

| Silicon atoms | ALCHEMI whole workflow | LAMMPS whole workflow | ALCHEMI MD call | LAMMPS MD loop |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 17.572 | 91.641 | 4.914 | 75.824 |
| 512 | 17.705 | 87.564 | 5.099 | 71.741 |
| 8,000 | 37.821 | 106.053 | 24.103 | 88.782 |
| 32,768 | 110.429 | 264.351 | 92.750 | 237.431 |

At 32,768 atoms, these short runs take about 110 seconds with ALCHEMI
and 264 seconds with LAMMPS. That includes startup; it is not a GPU-kernel
speedup. Whole-workflow repeat ranges at this size were 110.426–110.531
and 263.872–266.540 seconds, respectively.
````

````{tab-item} JUPITER
:sync: jupiter

Median seconds (minimum–maximum) on one JUPITER GH200 GPU:

| Silicon atoms | ALCHEMI whole workflow | LAMMPS whole workflow | ALCHEMI MD call | LAMMPS MD loop |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 21.825 (19.582–24.136) | 71.367 (68.150–640.335) | 4.578 | 54.032 |
| 512 | 22.083 (20.271–247.676) | 71.964 (71.911–344.840) | 5.442 | 54.336 |
| 8,000 | 41.490 (41.118–346.352) | 104.141 (103.897–355.570) | 24.017 | 84.682 |
| 32,768 | 114.314 (114.258–343.848) | 265.285 (261.875–455.189) | 92.843 | 233.106 |

```{warning}
JUPITER whole-workflow times varied widely, with unexplained delays outside
measured MD. Every repeat is retained. These ranges do not support a stable
engine ranking or a hardware comparison with Arrhenius.
```
````
`````

```{note}
32,768 atoms is a completed test size, not a GPU-memory maximum. At this
size the final potential energies differed by about 0.00335 eV/atom after
21 fs on both sites. The difference is unresolved: matched timing inputs
do not establish scientific equivalence.
```

## Independent trajectories: increase the batch size

ALCHEMI batches systems in one process. Ordinary LAMMPS sharing starts a
separate process for each trajectory; CUDA MPS changes how those processes
share the GPU. All three methods below use **one GPU** and complete the
same number of measured steps per trajectory.

`````{tab-set}
````{tab-item} Arrhenius
:sync: arrhenius

Median whole-workflow seconds on one Arrhenius GH200 GPU:

| Atoms per trajectory | Trajectories | ALCHEMI batch | LAMMPS ordinary sharing | LAMMPS CUDA MPS |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 17.572 | 91.641 | not tested |
| 64 | 2 | 17.685 | 88.074 | 88.970 |
| 64 | 4 | 17.653 | 87.157 | 84.972 |
| 64 | 8 | 17.630 | 96.970 | 84.920 |
| 512 | 1 | 17.705 | 87.564 | not tested |
| 512 | 2 | 17.864 | 89.947 | 85.911 |
| 512 | 4 | 20.320 | 87.479 | 88.352 |
| 512 | 8 | 26.350 | 109.140 | 89.170 |

For 64 atoms, the ALCHEMI batch completes eight trajectories in roughly
the same time as one. Its total completed work therefore increases much
faster than its elapsed time. Eight trajectories gave the highest
throughput among the sizes tested here—not an optimum batch size or proof
of full GPU occupancy.
````

````{tab-item} JUPITER
:sync: jupiter

Median whole-workflow seconds (minimum–maximum) on one JUPITER GH200 GPU:

| Atoms per trajectory | Trajectories | ALCHEMI batch | LAMMPS ordinary sharing | LAMMPS CUDA MPS |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 21.825 (19.582–24.136) | 71.367 (68.150–640.335) | not tested |
| 64 | 2 | 21.544 (21.433–54.473) | 78.870 (73.853–79.272) | 303.420 (293.801–350.874) |
| 64 | 4 | 19.840 (19.744–20.039) | 86.297 (84.569–87.440) | 305.000 (69.275–582.043) |
| 64 | 8 | 236.323 (22.822–293.039) | 378.148 (369.871–653.521) | 315.483 (79.394–377.167) |
| 512 | 1 | 22.083 (20.271–247.676) | 71.964 (71.911–344.840) | not tested |
| 512 | 2 | 20.337 (20.161–20.768) | 78.725 (78.519–79.712) | 293.680 (65.714–300.294) |
| 512 | 4 | 23.975 (23.802–23.995) | 86.214 (85.304–86.248) | 356.288 (303.472–596.110) |
| 512 | 8 | 237.613 (220.790–263.248) | 375.344 (114.113–586.199) | 344.726 (78.461–578.893) |

```{warning}
The MPS and eight-trajectory points used separate allocations. Large repeat
variation prevents choosing an optimum batch size or attributing a change
to MPS alone. Compare the ranges, not just the median.
```
````
`````

To compare completed work, calculate **completed replica-steps per second**:

```text
number of trajectories × measured steps per trajectory / whole-workflow seconds
```

For example, eight 64-atom ALCHEMI trajectories on Arrhenius complete
`8 × 200 = 1,600` measured replica-steps in 17.63 seconds:
`1,600 / 17.63 ≈ 90.8` replica-steps/s. This counts the work done by
all trajectories; it does not mean one trajectory advances at that rate.
Always show elapsed time as well as throughput.

## One large trajectory: share one GPU between MPI ranks

More MPI ranks do not add GPU memory when they share a single GPU.
On Arrhenius, a 39,304-atom LAMMPS trajectory completed with one, two
and four ranks under CUDA MPS. The next checked sizes ran out of memory:

| MPI ranks on one GPU | Completed size | Next checked failure |
| ---: | ---: | --- |
| 1 | 39,304 atoms | 46,656 atoms: CUDA OOM |
| 2 | 39,304 atoms | 46,656 atoms: CUDA OOM |
| 4 | 39,304 atoms | 64,000 atoms: CUDA OOM; 46,656 untested |

These short capacity checks do not locate the exact memory limit. A
separate timing run measured 200 steps after ten warm-up steps, three times:

| MPI ranks on one GPU | Median LAMMPS MD-loop seconds | Throughput gain over one rank |
| ---: | ---: | ---: |
| 1 | 275.975 | — |
| 2 | 266.009 | 3.75% |
| 4 | 266.574 | 3.53% |

Two ranks helped modestly; four did not improve on two. This is a
within-LAMMPS result for one input and GPU, not the cross-engine NVE
comparison above. [Methods and checks](../reference/benchmarks.md)
retains the individual repeats, capacity-test details and evidence records.
