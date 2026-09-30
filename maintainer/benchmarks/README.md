# Matched MD benchmark contract

These are maintainer qualification sources, not participant notebook cells.
The simple NVT examples in `examples/` remain runnable demonstrations; do
not compare their elapsed times as an engine speedup.

## Common starting state

`generate_starts.py` creates one immutable pair of starting files per replica:
JSON for ALCHEMI and LAMMPS `read_data` text. Both carry the same silicon
positions, periodic cell, and zero-centre-of-mass thermal velocities. The
JSON velocities are in ALCHEMI's internal units; the data-file velocities
are in LAMMPS `metal` Å/ps. The conversion uses
`FS_PER_INTERNAL_TIME=10.180505710759414`, independently checked in the
pinned `nvalchemi-toolkit==0.2.0` wheel's `dynamics/_units.py`. Every generated
file gets a digest in `manifest.json`. The starting files are private run
artifacts, never committed to the lesson.

Generate a new 64-atom, eight-replica start directory on a writable private
filesystem with:

```bash
python maintainer/benchmarks/generate_starts.py \
  --cells 2 --replicas 8 --output-dir "$PRIVATE_RESULTS/starts-64-8"
```

The intended comparison is NVE velocity-Verlet at 0.1 fs with the same
original MACE-MP-0a small checkpoint and its verified ML-IAP export. ALCHEMI
uses `run_alchemi_nve.py`; LAMMPS uses `lammps_mace_nve.in`. Both start from
the generated records. Keep initial geometry, velocities, model, atom count,
warm-up, measured steps, GPU type and allocation fixed within each table.
Only method or the declared sweep dimension may change. Check complete
replica count and finite outputs before accepting a row.

## Measurement boundaries

For each method, time *whole completion of the requested trajectories* from
client launch to clean exit. This is the primary cross-engine comparison and
includes import/model/setup overhead. Also record ALCHEMI's synchronized
measured-MD duration and each LAMMPS replica's measured `Loop time`.
LAMMPS's longest replica loop is **not** a synchronized eight-replica MD wall
clock; do not divide it by ALCHEMI's batch MD time to claim a kernel speedup.
The summarizer refuses a missing, failed, or step-mismatched LAMMPS replica.
Never mix whole-workflow and MD-only clocks in one ratio.

Use at least three counterbalanced rounds at each fixed workload; retain all
completed and failed case identities. A transport-uncertain Slurm submission
is reconciled read-only, not replayed. Arrhenius and JUPITER get separate
tables and separate artifact identities.

## Planned matrices

1. Eight trajectories on one GPU: 64 and 512 atoms each; ALCHEMI batch,
   LAMMPS ordinary sharing, and LAMMPS CUDA MPS. Ten warm-up and 200
   measured steps per trajectory.
2. One-trajectory size sweep: 64, 512, 8,000, and 32,768 atoms on one GPU.
   Do not include a size that either engine cannot complete in a speed ratio.
   Capacity brackets are functional results, not timings.
3. Batch-size sweep: 1, 2, 4, and 8 trajectories at fixed 64 and 512 atoms,
   extending only after a bounded GPU-memory preflight. Report completed
   replica-steps per second and total completion time.
4. LAMMPS within-engine scaling: one trajectory at fixed atom count across
   1, 2, and 4 GPUs (strong scaling), then atom counts approximately
   proportional to GPU count (weak scaling). Keep its model, starting-state
   generation, integrator, timestep, and measurement boundary fixed; record
   the exact atom count because cubic silicon cells give discrete sizes.

The first three matrices compare methods only at identical workload points.
The fourth varies resources by design and must never be presented as an
ALCHEMI multi-GPU result. Short functional checks and OOM probes are not
speed measurements. A result enters the lesson only after site qualification
and a separate science-evidence record in `mlip-hands-on`.
