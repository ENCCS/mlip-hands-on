# Read a benchmark result

A useful performance table names the machine and GPU, exact model,
simulation input, number of atoms and independent trajectories, number of
measured steps, warm-up, elapsed time, and software build. A throughput
number without those fields cannot be fairly compared or reproduced.

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

A fresh Arrhenius benchmark used this branch's ALCHEMI SIF example and
native LAMMPS CLI plus `.in` file—not the historical main-branch Python
driver. Each method finished eight independent silicon trajectories of
either 64 or 512 atoms on one GH200 GPU. Each trajectory advanced 200 steps
at 0.1 fs with the MACE-MP-0a small model. LAMMPS ran eight processes with
ordinary GPU sharing or CUDA MPS; ALCHEMI advanced an eight-system batch in
one process. No separate warm-up steps were used in this comparison.

| Atoms per trajectory | ALCHEMI batch | LAMMPS ordinary sharing | LAMMPS CUDA MPS |
| ---: | ---: | ---: | ---: |
| 64 | 17.79 s | 93.00 s | 86.02 s |
| 512 | 26.00 s | 104.84 s | 80.85 s |

These are **median whole-workflow times** for all eight trajectories to
finish, from three differently ordered rounds in Arrhenius job `3195110`.
They include client startup, model loading, initial setup, and MD; they
exclude queue time and the once-per-job native runtime extraction. At 64
atoms, the LAMMPS ordinary and MPS medians were 5.23 and 4.84 times the
ALCHEMI median; at 512 atoms, 4.03 and 3.11 times. Those ratios describe
the complete workflows, **not** the speed of a MACE force kernel or an
isolated MD step. All eighteen cases completed, but three rounds in one
allocation do not establish site-wide performance confidence.

Both routes used a Langevin thermostat, but ALCHEMI and LAMMPS do not have
identical integrator implementations or matched initial velocities. This is
a comparison of a stated workload, not evidence of trajectory equivalence.
The benchmark varied `cells=2` and `cells=4` in the [ALCHEMI example](../../examples/alchemi_si_one_cell.py)
and the [LAMMPS input](../../examples/lammps_mace.in); the usual notebook
demo remains at `cells=2`. The MLIP science project retains the exact
artifact hashes, all eighteen timings, and qualification limits.

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
