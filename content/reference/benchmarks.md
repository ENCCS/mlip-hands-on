# Benchmark methods and checks

The [result tables](../episodes/09-results.md) show elapsed time and its
meaning. This reference records how the measurements were made and what
was checked. Scientific contracts and original evidence belong to the
separate MLIP science repository; private logs and artifacts are not shipped
with this lesson.

## Matched NVE workloads

Both sites use MACE-MP-0a small, its float32 original checkpoint and ML-IAP
export, diamond silicon positions/cells, and generated zero-centre-of-mass
velocities. Both engines use NVE velocity-Verlet at 0.1 fs, ten warm-up
steps and 200 measured steps. Three rounds change the method/case order.
Matching weights and inputs does not make coordinate arithmetic or
trajectories bitwise identical.

Single-trajectory sizes are 64, 512, 8,000 and 32,768 atoms. Independent
trajectory counts are 1/2/4/8 at 64 and 512 atoms. MPS for a single
process was not tested. JUPITER allocates a full node for these tests;
single-GPU workloads still use only one of its GH200 GPUs.

The notebook Langevin demonstrations are functional examples, not these
NVE benchmarks. Historical Python-driver and unmatched Langevin timings
are not used as cross-engine speedups in the lesson.

## Timing boundaries

| Clock | What is included | How to use it |
| --- | --- | --- |
| Whole workflow | Launch, imports, model loading, setup, warm-up, MD, shutdown | Compare the cost of completing the same short workload |
| ALCHEMI MD call | Synchronized timed batch run after warm-up | Locate time spent in this engine's MD call |
| LAMMPS MD loop | Engine-reported measured loop after warm-up | Compare the same LAMMPS workload and scaling configuration |

Neither internal timer is a GPU-kernel timer. Independent LAMMPS loops
do not begin together; their longest loop is not synchronized batch elapsed
time. Do not divide it by the ALCHEMI timer to claim a kernel speedup.
Queue wait is excluded from every table.

For eight trajectories, median internal diagnostics at 64/512 atoms were:

| Site | ALCHEMI MD call | Longest ordinary LAMMPS loop | Longest MPS LAMMPS loop |
| --- | --- | --- | --- |
| Arrhenius | 4.95 / 13.31 s | See original replica logs | See original replica logs |
| JUPITER | 4.65 / 13.46 s | 69.38 / 85.83 s | 55.84 / 53.37 s |

All repeats are retained. JUPITER whole-workflow times varied widely,
and result-file reads intermittently timed out. Those observations do not
isolate the cause of every delay. They support neither a stable engine
ranking nor a hardware comparison with Arrhenius.

## Separate allocations and validation

On Arrhenius, the eight-trajectory comparison ran in one allocation;
the one/two/four-trajectory sweep ran in another. On JUPITER, ordinary
sharing and MPS used separate allocations; the eight-trajectory points
also used different allocations from the smaller sweep. Starting states
match, but cache and filesystem conditions need not.

Every site has a complete eighteen-case eight-trajectory matrix:
96 LAMMPS replica logs and six ALCHEMI completion records. Six MPS service
proofs were also checked for JUPITER. Each complete smaller size/batch
sweep has sixty cases, 84 LAMMPS logs and 24 ALCHEMI records; the JUPITER
MPS sweep has twelve observed-service proofs.

Checks cover declared atoms, replicas, ensemble, warm-up and measured
steps, finite final values, completion, log/summary consistency and timer
boundaries. Original per-allocation tables are retained unchanged, including
when a separate combined table is derived. Failed or partial attempts do
not enter the published matrices.

At eight trajectories, maximum corresponding final potential differences
were 0.0069 eV on Arrhenius and 0.0067 eV on JUPITER after 21 fs.
At 32,768 atoms, both sites show about 0.00335 eV/atom disagreement.
Initial forces/energies and stepwise drift need investigation before
claiming scientific equivalence. These checks do not establish stress,
NVT or long-trajectory agreement.

## One-node scaling

Strong scaling keeps 32,768 atoms fixed with one MPI rank per GPU.
Weak scaling uses 8,000 / 17,576 / 32,768 atoms on one / two / four GPUs,
or 8,000 / 8,788 / 8,192 atoms per GPU. Cubic supercells make this an
approximate constant-work-per-GPU experiment. Each site completed all
eighteen cases with the same NVE settings above.

Logs confirm requested MPI and Kokkos GPU counts and finite final output.
No independent device-identity or MPI-bandwidth probe was added to the
timed jobs. The tables establish neither inter-node performance nor a
VRAM maximum.

## Shared-GPU MPI and capacity checks

The separate Arrhenius 39,304-atom check used CUDA MPS and the native
ML-IAP/Kokkos input. Capacity cases used ten warm-up steps and two further
steps. At 46,656 atoms, one and two ranks failed with CUDA OOM during
the first force evaluation. Four ranks at that size were untested; their
64,000-atom attempt failed with CUDA OOM. Two failed multi-rank steps
required cancellation because Slurm did not close them promptly after OOM.
No failed case was replayed.

A later run measured 200 steps after ten warm-up steps. These are
within-LAMMPS timings, not the cross-engine NVE matrix:

| Round | 1 rank | 2 ranks | 4 ranks |
| ---: | ---: | ---: | ---: |
| 1 | 276.819 s | 266.362 s | 266.697 s |
| 2 | 274.817 s | 266.009 s | 266.574 s |
| 3 | 275.975 s | 265.571 s | 265.794 s |

## Functional notebook checks

One- and eight-trajectory ALCHEMI MD, four-structure FIRE relaxation,
LAMMPS CLI MD and CG relaxation, and two/four-rank MPI commands completed
on both sites. The full one-trajectory, batch and relaxation MyST notebook
pages were executed on both sites; the eight-process LAMMPS notebook route
also completed on both. Short functional checks are not warmed benchmarks.

JUPITER checks used both a native ALCHEMI environment and the same
ARM/GH200 SIF tested on Arrhenius. Rebuilding the current SIF definition
or native LAMMPS build script needs its own qualification; an existing
binary's successful run does not establish that a new build works.

On Arrhenius, direct batch-shell replica launches failed in MPICH/OFI
initialization; a shared PMI-2 step also failed when children sent PMI-1
initialization commands. The completed route wraps the independent
processes in one networked `--mpi=none` step. It is distinct from one
coupled MPI trajectory. Historical Python-driver measurements are retained
only in the science project's evidence.

## Evidence records

All jobs listed below completed with exit `0:0`. Files are under
`docs/evidence/` in the MLIP science repository and retain raw repeats,
artifact identities and failed-attempt history. Matched input generators,
runners and CPU validators are in this lesson's `maintainer/benchmarks/`;
participants do not need that layer to run the examples.

| Measurement | Site | Job IDs | Evidence filename |
| --- | --- | --- | --- |
| Eight trajectories | Arrhenius | 3197862 | `arrhenius-matched-nve-eight-2026-10-01.md` |
| Size/batch sweep | Arrhenius | 3203214 | `arrhenius-matched-nve-sweeps-2026-10-01.md` |
| Strong/weak scaling | Arrhenius | 3203275 | `arrhenius-matched-nve-scaling-2026-10-01.md` |
| Eight trajectories | JUPITER | 2127710, 2127664 | `jupiter-matched-nve-eight-2026-10-01.md` |
| Size/batch sweep | JUPITER | 2127712, 2127714 | `jupiter-matched-nve-sweeps-2026-10-01.md` |
| Strong/weak scaling | JUPITER | 2127713 | `jupiter-matched-nve-scaling-2026-10-01.md` |
| Shared-GPU MPI timing | Arrhenius | 3180940 | `arrhenius-one-gpu-mpi-200-2026-09-30.md` |
