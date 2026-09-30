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

The cell below uses that launcher. This is process sharing, not ALCHEMI's
in-process batch. It does not create eight overlapping Slurm steps. The native
CLI route has passed on JUPITER Booster. On Arrhenius, separate processes
fail during MPICH/OFI initialization, while an MPI-partition alternative
aborts during shutdown; this branch does not present either as a working
Arrhenius exercise.

```{code-cell} ipython3
%%bash
cd ../..
source "scripts/${MLIP_SITE}-lammps-env.sh"
if [[ "$MLIP_SITE" == arrhenius ]]; then
  echo 'SKIPPED: eight-process native LAMMPS is not qualified on Arrhenius'
  exit 0
fi
time bash scripts/run-lammps-replicas.sh "$PWD" "$MLIP_LMP" \
  "$MLIP_MLIAP_MODEL" "$MLIP_ARTIFACT_ROOT/replicas-${SLURM_JOB_ID}" 200
```

The output directory must be new. Do not treat eight independent process
times as one elapsed time. CUDA MPS is an optional third configuration;
enabling it changes GPU process scheduling but does not turn LAMMPS into an
ALCHEMI batch.

These 200-step cells are a short functional demonstration. To measure
throughput, run a separate declared workload with a warm-up interval and
more measured steps; pass the measured step count as the launcher's final
argument.

The [reference](../reference/limits.md) distinguishes historical process
measurements from runs qualified with this branch's CLI input. Only compare
numbers measured with the same model, atom count, steps, warm-up convention,
and GPU allocation.
