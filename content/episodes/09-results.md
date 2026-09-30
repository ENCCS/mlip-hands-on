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

No new CLI benchmark table is published yet. Run a declared measurement
workload before quoting speed or scaling. Include failed attempts in private
qualification evidence instead of silently treating a corrected rerun as
the original result.

For one trajectory, report MD steps per second. For independent trajectories,
report **completed replica-steps per second**—the sum of completed steps over
all replicas divided by total elapsed wall time. Also show elapsed time so
readers can judge the actual cost of a session.
