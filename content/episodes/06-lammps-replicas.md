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

# Run LAMMPS replicas on one GPU

LAMMPS can run independent trajectories as separate processes sharing
a GPU. This is not ALCHEMI's in-process batching: each process loads a
model and has its own GPU context. The [input file](../../examples/lammps_mace.in)
creates the same 64-atom silicon cell for each process.

First run one process on an allocated GPU. The input is the same
[`lammps_mace.in`](../../examples/lammps_mace.in) used in the previous
episode; `-var` supplies the model, cell count and seeds.

```{code-cell} ipython3
%%bash
cd ../..
source "scripts/${MLIP_SITE}-lammps-env.sh"
launch=()
if [[ "$MLIP_SITE" == arrhenius ]]; then
  launch=(srun --mpi=pmi2 --nodes=1 --ntasks=1 --gpus=1)
fi
time "${launch[@]}" "$MLIP_LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -log none -in examples/lammps_mace.in \
  -var model "$MLIP_MLIAP_MODEL" -var cells 2 -var warmup 10 \
  -var steps 200 -var seed 20260924 -var bath_seed 20260925
```

For eight independent simulations, the
[launcher](../../scripts/run-lammps-replicas.sh) starts eight processes
inside the existing allocation. Each gets different velocity and thermostat
seeds and writes a separate log. It waits for every process to exit:

```{literalinclude} ../../scripts/run-lammps-replicas.sh
:language: bash
:linenos:
```

The cell uses one launcher for all eight processes, not eight Slurm steps.
Arrhenius needs a networked wrapper step for native LAMMPS; JUPITER can
start the processes from the notebook's batch shell.

```{note}
The Arrhenius `--mpi=none` option belongs to the **one-rank wrapper step**.
The LAMMPS build still supports MPI, but these are eight independent
trajectories—not ranks collaborating on one trajectory.
```

```{code-cell} ipython3
%%bash
cd ../..
source "scripts/${MLIP_SITE}-lammps-env.sh"
launch=()
if [[ "$MLIP_SITE" == arrhenius ]]; then
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  launch=(srun --mpi=none --nodes=1 --ntasks=1 --gpus=1 \
    --cpus-per-task=8 --cpu-bind=cores)
fi
time "${launch[@]}" bash scripts/run-lammps-replicas.sh "$PWD" "$MLIP_LMP" \
  "$MLIP_MLIAP_MODEL" "$MLIP_ARTIFACT_ROOT/replicas-${SLURM_JOB_ID}" 200
```

Read each trajectory's log in the new output directory. To repeat the
example within the same job, choose a different output directory. The
outer `time` measures how long **all eight** processes take to finish;
one process's MD-loop time does not measure the entire set.

The launcher's fifth argument sets steps, its sixth sets `cells`
(`2` gives 64 atoms; `4` gives 512), and its seventh sets warm-up steps.
This example uses 200 measured steps after ten warm-up steps.

CUDA MPS is an optional alternative: it changes how CUDA processes share
the GPU, not how the systems are batched inside LAMMPS. The
[result tables](09-results.md) compare ordinary sharing and MPS using
a separate matched NVE workload.
