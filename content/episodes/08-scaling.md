# One trajectory across GPUs

The earlier examples use one GPU. For a larger *single* LAMMPS trajectory,
MPI ranks can divide the simulation domain, with one GPU per rank. The
[input file](../../examples/lammps_mace.in) accepts `cells`: a diamond
supercell has `8 × cells³` atoms. The
[MPI command](../../scripts/run-lammps-mpi.sh) launches the same input with
`srun` from an existing allocation.

On one node with four allocated GPUs, the launcher amounts to this pattern.
`-k on g 4` is the number of GPUs per node, not per rank; keeping a common
visible GPU set for all ranks avoids per-rank GPU-mask mismatches:

```bash
export MLIP_ALLOCATED_CUDA_DEVICES="$CUDA_VISIBLE_DEVICES"
srun --ntasks=4 --gpu-bind=none /bin/bash -c \
  'export CUDA_VISIBLE_DEVICES="$MLIP_ALLOCATED_CUDA_DEVICES"; exec "$@"' _ \
  "$MLIP_LMP" -k on g 4 -sf kk \
  -pk kokkos newton on neigh half gpu/aware on \
  -log none -in examples/lammps_mace.in \
  -var model "$MLIP_MLIAP_MODEL" -var cells 8 -var warmup 0 \
  -var steps 100 -var seed 20260924 -var bath_seed 20260925
```

Add your site's required `--mpi` setting to `srun`; Arrhenius's tested
MPICH route uses `--mpi=pmi2`. The supplied launcher reads `MLIP_SRUN_MPI`
for this and checks the allocation boundary before calling `srun`.

For strong scaling, fix `cells` and vary GPU count. For weak scaling,
increase `cells` with GPU count so atoms per GPU stay approximately fixed.
Record the actual atoms, ranks, nodes, warm-up, measured steps, elapsed
seconds, and GPU type alongside every result. A calculation that merely
finishes is not a scaling measurement.

```{warning}
Do not launch this from a login node. Request an allocation for the chosen
site first. Multi-node runs may require a different partition or MPI/GPU
transport configuration than a one-node example. The Arrhenius short-test
reservation must be checked for the requested node count; `--test-only`
does not establish that the job will run promptly.
```

This episode focuses on the LAMMPS MPI route. The ALCHEMI one-GPU batch
demonstration is not a claim of multi-GPU domain decomposition.

## Measured one-node scaling on Arrhenius

A separate matched-NVE benchmark used MACE-MP-0a small, silicon starting
structures and velocities from the common generator, a 0.1 fs timestep,
ten warm-up steps and 200 measured steps. Unlike the short notebook demo
above, these runs hold the integrator and measurement boundary fixed.
Three rounds varied the case order. All 18 cases completed on one Arrhenius
node; times below are median LAMMPS **MD-loop seconds**, excluding model
loading, warm-up and queue wait.

### Strong scaling: keep 32,768 atoms fixed

| GH200 GPUs | MPI ranks | MD-loop seconds | Speedup over one GPU | Parallel efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 237.285 | 1.00× | 100% |
| 2 | 2 | 139.269 | 1.70× | 85.2% |
| 4 | 4 | 89.598 | 2.65× | 66.2% |

Four GPUs shorten this trajectory's measured MD time, but do not deliver
four times the throughput. The ratio is within LAMMPS, not an ALCHEMI
comparison. It also does not include the cost of launching a short session.

### Weak scaling: increase atoms with GPU count

| GH200 GPUs | Total atoms | Atoms per GPU | MD-loop seconds |
| ---: | ---: | ---: | ---: |
| 1 | 8,000 | 8,000 | 87.845 |
| 2 | 17,576 | 8,788 | 93.184 |
| 4 | 32,768 | 8,192 | 89.429 |

The MD time stays within 6.1% of the one-GPU value while the total system
grows. This is **approximate** weak scaling: cubic diamond supercells give
discrete sizes, so atoms per GPU are not exactly constant.

```{note}
These are one-node Arrhenius results for the pinned model and native build.
They do not establish a VRAM maximum or multi-node performance. The logs
confirm the requested MPI and Kokkos GPU counts; this benchmark did not
add an independent device-identity or MPI-bandwidth probe.
```

Job `3203275` completed with exit `0:0`. Every round and the artifact
identities are recorded in the MLIP science project at
`docs/evidence/arrhenius-matched-nve-scaling-2026-10-01.md`. The matched
inputs and runners are retained in `maintainer/benchmarks/`.

## Can more MPI ranks enlarge a one-GPU simulation?

Two or four MPI ranks can divide one trajectory while sharing a single GPU.
That does **not** add GPU memory: each rank creates its own GPU context and
loads the model. More ranks may therefore *lower* the largest atom count that
fits. Test this separately from the one-rank-per-GPU scaling above.

In an existing **one-GPU** allocation with CUDA MPS enabled, use the same
input and `cells` value for each rank count. For example, `cells=16` creates
32,768 silicon atoms:

```bash
source scripts/arrhenius-lammps-env.sh
bash scripts/run-lammps.sh "$PWD" "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 16 100
bash scripts/run-lammps-shared-gpu-mpi.sh "$PWD" 2 "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 16 100
bash scripts/run-lammps-shared-gpu-mpi.sh "$PWD" 4 "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 16 100
```

The [shared-GPU launcher](../../scripts/run-lammps-shared-gpu-mpi.sh) uses
`-k on g 1`: **one GPU per node**, not one per rank. On Arrhenius, request
`--network=single_node_vni` and use the site's qualified PMI-2 route.
Increase `cells` in fresh, bounded runs; record completed sizes and explicit
CUDA out-of-memory failures separately. A short completed run establishes a
capacity point, not sustained throughput or scientific equivalence. CUDA MPS
is recommended for usable multi-rank Kokkos performance; without it, do not
interpret a slow shared-GPU run as a meaningful speed comparison.
The [bounded Arrhenius result](09-results.md)
reports the completed sizes and a separate 200-step timing check; neither
establishes an absolute memory limit or a universal speedup.
