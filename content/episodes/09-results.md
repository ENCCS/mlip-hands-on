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
| LAMMPS CLI, eight independent GPU-sharing processes (this branch) | not qualified | completed |

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
declared. The eight-replica LAMMPS notebook route completed on JUPITER but
remains unqualified on Arrhenius.

On Arrhenius, eight separate native LAMMPS processes failed during MPICH/OFI
initialization. An eight-partition MPI route ran the MD steps but aborted
during shutdown. Neither is a completed participant exercise, and the
historical Python-route result in the table above does not qualify this
branch's CLI route.

No new CLI benchmark table is published yet. Run a declared measurement
workload before quoting speed or scaling. Include failed attempts in private
qualification evidence instead of silently treating a corrected rerun as
the original result.

For one trajectory, report MD steps per second. For independent trajectories,
report **completed replica-steps per second**—the sum of completed steps over
all replicas divided by total elapsed wall time. Also show elapsed time so
readers can judge the actual cost of a session.
