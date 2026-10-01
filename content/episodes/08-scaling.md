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
Record atom count, GPU count and MD time for each run. Keep the model,
timestep, warm-up and measured step count the same when comparing runs.

```{warning}
Do not launch this from a login node. Request an allocation for the chosen
site first. Multi-node runs may require a different partition or MPI/GPU
transport configuration than a one-node example. The Arrhenius short-test
reservation must be checked for the requested node count; `--test-only`
does not establish that the job will run promptly.
```

Here, MPI ranks work on different parts of **one trajectory**. In the
ALCHEMI batch example, the trajectories are independent instead.

## Measured one-node scaling

These measurements use silicon and MACE-MP-0a small, NVE at 0.1 fs,
ten warm-up steps and 200 measured steps. Each value is the median of
three runs. **MD-loop seconds** exclude loading, warm-up and queue wait;
they measure the time LAMMPS spends advancing the trajectory.

`````{tab-set}
````{tab-item} Arrhenius
:sync: arrhenius

### Strong scaling: keep 32,768 atoms fixed

| GH200 GPUs | MPI ranks | MD-loop seconds | Speedup over one GPU | Parallel efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 237.285 | 1.00× | 100% |
| 2 | 2 | 139.269 | 1.70× | 85.2% |
| 4 | 4 | 89.598 | 2.65× | 66.2% |

Four GPUs reduce MD time from 237 to 90 seconds, a 2.65× speedup.
The gain is less than fourfold; adding GPUs does not divide the time exactly.

### Weak scaling: increase atoms with GPU count

| GH200 GPUs | Total atoms | Atoms per GPU | MD-loop seconds |
| ---: | ---: | ---: | ---: |
| 1 | 8,000 | 8,000 | 87.845 |
| 2 | 17,576 | 8,788 | 93.184 |
| 4 | 32,768 | 8,192 | 89.429 |

The MD time stays within 6.1% of the one-GPU value while the total system
grows. This is **approximate** weak scaling: cubic diamond supercells give
discrete sizes, so atoms per GPU are not exactly constant.

````

````{tab-item} JUPITER
:sync: jupiter

### Strong scaling: keep 32,768 atoms fixed

| GH200 GPUs | MPI ranks | MD-loop seconds | Speedup over one GPU | Parallel efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 235.822 | 1.00× | 100% |
| 2 | 2 | 136.923 | 1.72× | 86.1% |
| 4 | 4 | 87.376 | 2.70× | 67.5% |

Four GPUs reduce MD time from 236 to 87 seconds, a 2.70× speedup.
This compares the same LAMMPS trajectory, not separate independent runs.

### Weak scaling: increase atoms with GPU count

| GH200 GPUs | Total atoms | Atoms per GPU | MD-loop seconds |
| ---: | ---: | ---: | ---: |
| 1 | 8,000 | 8,000 | 84.763 |
| 2 | 17,576 | 8,788 | 91.825 |
| 4 | 32,768 | 8,192 | 87.122 |

Median MD time stays within 8.4% of the one-GPU value. The diamond
supercells make this approximate weak scaling, not exactly equal atoms
per GPU.

````
`````

```{note}
These are one-node, within-LAMMPS MD timings—not multi-node performance,
memory limits or ALCHEMI speedups. [Benchmark methods and checks](../reference/benchmarks.md)
records the repeats, software identities and validation details.
```

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
Increase `cells` gradually and record both completed sizes and CUDA
out-of-memory errors. Keep CUDA MPS enabled for this shared-GPU comparison.
The [Arrhenius results](09-results.md) show that adding ranks did not enlarge
the completed system in the sizes tested, and improved MD time only modestly.
