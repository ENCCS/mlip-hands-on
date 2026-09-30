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
in-process batch. It does not create eight overlapping Slurm steps. On
Arrhenius the MPI-enabled LAMMPS children need a networked Slurm step around
their launcher, without making the eight independent clients members of one
PMI job; JUPITER can launch them from the batch shell.

```{note}
The Arrhenius `--mpi=none` option belongs to the **one-rank wrapper step**.
It does not make this a serial LAMMPS build or turn eight trajectories into
one MPI trajectory. This route was functionally checked for eight 64-atom
replicas on one GH200; the multi-GPU MPI route is a separate exercise.
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

The output directory must be new. Do not treat eight independent process
times as one elapsed time. CUDA MPS is an optional third configuration;
enabling it changes GPU process scheduling but does not turn LAMMPS into an
ALCHEMI batch.

These 200-step notebook cells are a short functional demonstration. A
benchmark should declare its clock, workload, and repetition order before
running. The launcher's fifth argument sets MD steps, its sixth sets
`cells` (`2` is 64 atoms; `4` is 512), and its seventh sets warm-up steps. The
[completed eight-trajectory benchmark](09-results.md) used 200 steps and
zero separate warm-up steps with both engines; this notebook demonstration
retains its default ten-step LAMMPS warm-up.

The [reference](../reference/limits.md) distinguishes historical process
measurements from runs qualified with this branch's CLI input. Only compare
numbers measured with the same model, atom count, steps, warm-up convention,
and GPU allocation.
