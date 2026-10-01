# Read a benchmark result

A useful performance table names the machine and GPU, exact model,
simulation input, number of atoms and independent trajectories, number of
measured steps, warm-up, elapsed time, and software build. For a comparison,
also hold the starting structures and velocities, timestep, ensemble,
integrator, and thermostat parameters fixed wherever the engines permit it.
State any implementation difference that cannot be removed. Compare the same
clock boundary: whole workflow with whole workflow, or measured MD with
measured MD. A throughput number without this contract cannot be fairly
compared or reproduced.

The main-branch historical measurements used a Python LAMMPS driver. This
branch changes LAMMPS to a CLI plus `.in` file. Those historical numbers are
not measurements of the new route.

## Short functional checks

The current participant examples have been run with the pinned MACE-MP-0a
small model on allocated GH200 nodes. These checks establish that the
commands complete, not a performance ranking:

| Example | Arrhenius | JUPITER Booster |
| --- | --- | --- |
| ALCHEMI, one and eight 64-atom trajectories | completed | completed |
| ALCHEMI, four-structure FIRE relaxation | completed | completed |
| LAMMPS CLI, one 64-atom trajectory | completed | completed |
| LAMMPS CLI, Kokkos CG relaxation | completed | completed |
| LAMMPS CLI, two and four MPI ranks on one node | completed | completed |
| LAMMPS, eight independent GPU-sharing processes (historical Python route) | completed | not yet run |
| LAMMPS CLI, eight independent GPU-sharing processes (this branch) | completed | completed |

The JUPITER ALCHEMI checks used both a native environment and the same
hash-verified ARM/GH200 SIF previously run on Arrhenius. Rebuilding the
current SIF definition or native LAMMPS build script remains a separate
qualification. The short checks used five MD steps for the cross-site
functional gate; they do not measure sustained throughput.

The full MyST notebook pages for [one silicon trajectory](04-silicon-md.md)
and [batched ALCHEMI trajectories](05-batched-md.md) were executed on both
sites. The [relaxation page](07-relaxation.md) was also executed on both
sites. These were functional runs, not comparable
timings: the integrators differ and no warmed measurement interval was
declared. The eight-replica LAMMPS notebook route completed on both sites.

On Arrhenius, launching eight separate native LAMMPS processes directly from
the batch shell failed during MPICH/OFI initialization. A shared PMI-2 step
also failed because the children sent PMI-1 initialization commands. The
completed route instead places the launcher inside one networked Slurm step
with `--mpi=none`, leaving the children to initialize independently. Earlier
failed attempts are not counted as completed runs. These short functional
checks do not establish performance or scientific equivalence.

## Eight independent trajectories on one GPU

The [ALCHEMI batch](05-batched-md.md) and [LAMMPS process-sharing](06-lammps-replicas.md)
examples both complete eight trajectories. They are runnable demonstrations,
but their default Langevin integrators differ and their initial velocities
are not matched. Do **not** divide their elapsed times to claim a speedup.
The earlier Arrhenius 64- and 512-atom workflow timings remain in the MLIP
science project's evidence as an exploratory functional measurement, not an
accepted cross-engine benchmark for this lesson.

### Matched NVE comparison: Arrhenius

A separate benchmark completed on **one Arrhenius GH200 GPU**. Both engines
used the same MACE-MP-0a small model (original checkpoint and ML-IAP export),
silicon positions, periodic cells and initial velocities, NVE velocity-Verlet
at 0.1 fs, ten warm-up steps and 200 measured steps per trajectory. Three
rounds varied the method order within one allocation. Every trajectory
completed; the notebook's default Langevin examples are not these inputs.

| Atoms per trajectory | Independent trajectories | ALCHEMI batch | LAMMPS ordinary sharing | LAMMPS CUDA MPS |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 8 | 17.63 s | 96.97 s | 84.92 s |
| 512 | 8 | 26.35 s | 109.14 s | 89.17 s |

These are median **whole-workflow** times: launch to completion of all eight
trajectories, including imports, model loading, setup, warm-up and MD.
For example, the 64-atom ALCHEMI row completes 1,600 measured replica-steps
in 17.63 s: about 90.8 completed replica-steps per whole-workflow second.
This describes the cost of these short runs, not a GPU-kernel speedup.

ALCHEMI's separately measured batch MD medians were 4.95 s (64 atoms) and
13.31 s (512 atoms). LAMMPS records an MD-loop time for each process, but
those intervals do not start together; the longest loop is not a comparable
synchronized batch clock. Do not use it to compute a cross-engine MD ratio.

```{note}
All 96 LAMMPS replica logs and six ALCHEMI batches passed completion and
finite-output checks. The largest final potential-energy difference between
corresponding trajectories was 0.0069 eV after 21 fs. This short sanity check
does not establish long-trajectory or NVT agreement. These are Arrhenius
results only; the matched JUPITER matrix is still being qualified.
```

The maintainer sources in `maintainer/benchmarks/` retain the matched starting
state generator and NVE runners. The MLIP science project records all three
rounds and artifact identities in
`docs/evidence/arrhenius-matched-nve-eight-2026-10-01.md`; job `3197862`
completed with exit `0:0`.

## One trajectory: increase the atom count

The matched NVE size sweep also completed on **one Arrhenius GH200 GPU**.
It uses the same model, starting-state generator, 0.1 fs timestep, ten
warm-up steps and 200 measured steps. Each size has three complete rounds.
Both model paths use float32 weights; this does not make the engines'
coordinate arithmetic or trajectories bitwise identical.

| Silicon atoms | ALCHEMI whole workflow | LAMMPS whole workflow | ALCHEMI measured MD call | LAMMPS MD loop |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 17.572 s | 91.641 s | 4.914 s | 75.824 s |
| 512 | 17.705 s | 87.564 s | 5.099 s | 71.741 s |
| 8,000 | 37.821 s | 106.053 s | 24.103 s | 88.782 s |
| 32,768 | 110.429 s | 264.351 s | 92.750 s | 237.431 s |

Compare the two **whole-workflow** columns: both measure launch to clean
completion. The MD columns help locate time spent in the simulation, but
ALCHEMI times a synchronized run call and LAMMPS reports its engine loop.
Their definitions differ; these columns are not a GPU-kernel speedup.
For 32,768 atoms, whole-workflow times ranged from 110.426 to 110.531 s
for ALCHEMI and 263.872 to 266.540 s for LAMMPS.

```{note}
32,768 atoms is the largest shared completed point in this sweep, not the
maximum size either engine can fit in GPU memory. No failed capacity point
was tested in this matched sweep.
```

## Increase the number of independent trajectories

Keep the size of each trajectory fixed and vary how many run together.
The table gives median **whole-workflow seconds** for all requested
trajectories to finish on one Arrhenius GH200 GPU.

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

Calculate completed replica-steps per second as
`trajectories × 200 / whole-workflow seconds`. For 64 atoms, eight
trajectories give 90.8 for ALCHEMI, 16.5 for ordinary LAMMPS sharing and
18.8 for MPS. For 512 atoms, the corresponding rates are 60.7, 14.7 and
17.9. Eight trajectories was the highest-throughput point **tested here**,
not an optimum batch size or proof that the GPU was fully occupied.

The one-, two- and four-trajectory points completed in job `3203214`;
the eight-trajectory points come from job `3197862`. Their starting files
match exactly, but the two allocations need not have identical cache or
filesystem conditions. All 60 new cases passed completion checks, including
84 LAMMPS replica logs and 24 ALCHEMI records. Full repeat values and
artifact identities are recorded in the MLIP science project at
`docs/evidence/arrhenius-matched-nve-sweeps-2026-10-01.md`.

### One trajectory, multiple MPI ranks sharing one GPU

An Arrhenius GH200 capacity check on 30 September 2026 used the pinned
MACE-MP-0a small model and the native ML-IAP/Kokkos input. CUDA MPS was
enabled. The same 39,304-atom trajectory completed with one, two, and four
MPI ranks sharing **one** GPU; the completed cases each ran ten warm-up and
two further MD steps. At 46,656 atoms, the one- and two-rank attempts failed
with explicit CUDA out-of-memory errors during the first force evaluation.
The four-rank 46,656-atom case was not tested. Its 64,000-atom attempt also
failed with CUDA OOM.

| MPI ranks on one GPU | Largest completed size in this check | Next checked failure |
| ---: | ---: | --- |
| 1 | 39,304 atoms | 46,656 atoms: CUDA OOM |
| 2 | 39,304 atoms | 46,656 atoms: CUDA OOM |
| 4 | 39,304 atoms | 64,000 atoms: CUDA OOM; 46,656 untested |

These are bounded observations, **not** the exact maximum atom count or a
performance comparison. Adding ranks did not raise the completed size in
this check. Two failed multi-rank steps required cancellation after one rank
reported OOM and Slurm did not close the step promptly. No failed case was
replayed. A larger single trajectory should instead be tested across more
GPUs, with one rank per GPU.

### Measured one-GPU MD-loop time

A later Arrhenius run measured the same 39,304-atom silicon trajectory with
native LAMMPS ML-IAP/Kokkos, MACE-MP-0a small, and CUDA MPS. One, two, or
four MPI ranks shared **one GH200 GPU**. Each case had ten warm-up steps and
200 measured MD steps. Three rounds changed the rank order to reduce simple
order effects. The table reports LAMMPS `Loop time` for the 200-step run;
it excludes model loading, warm-up, and scheduler wait.

| Round | 1 rank | 2 ranks | 4 ranks |
| ---: | ---: | ---: | ---: |
| 1 | 276.819 s | 266.362 s | 266.697 s |
| 2 | 274.817 s | 266.009 s | 266.574 s |
| 3 | 275.975 s | 265.571 s | 265.794 s |
| Median | 275.975 s | 266.009 s | 266.574 s |

For this workload, two ranks delivered about **3.75% more measured MD-step
throughput** than one rank; four delivered about **3.53% more**. Four ranks
did not improve on two. All nine 200-step runs completed in Arrhenius job
`3180940`. These are timings for one model, input, GPU, and site, not a
general rule for choosing MPI ranks. The earlier two-step capacity check
above is not used as speed evidence. The exact run identities and limits are
retained in the MLIP project's scientific evidence record.

Use a declared measurement workload before quoting other speed or scaling
results. Include failed attempts in private qualification evidence instead
of silently treating a corrected rerun as the original result.

For one trajectory, report MD steps per second. For independent trajectories,
report **completed replica-steps per second**—the sum of completed steps over
all replicas divided by total elapsed wall time. Also show elapsed time so
readers can judge the actual cost of a session.
