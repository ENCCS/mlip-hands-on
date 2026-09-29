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

LAMMPS can run independent trajectories as separate processes sharing a
GPU. This is not ALCHEMI's in-process batching: each process loads a model
and has its own GPU context. The [input file](../../examples/lammps_mace.in)
creates the same 64-atom silicon cell for every process. Change the two
random seeds for each replica.

First run one process on an allocated GPU:

```{code-cell} ipython3
%%bash
cd ../..
source "scripts/${MLIP_SITE}-lammps-env.sh"
time bash scripts/run-lammps.sh "$PWD" "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 2 2000
```

For eight processes, the small
[shell loop](../../scripts/run-lammps-replicas.sh) uses distinct seeds and
separate logs, then waits for every exit status:

```{code-cell} ipython3
%%bash
cd ../..
source "scripts/${MLIP_SITE}-lammps-env.sh"
time bash scripts/run-lammps-replicas.sh "$PWD" "$MLIP_LMP" \
  "$MLIP_MLIAP_MODEL" 8 "$MLIP_ARTIFACT_ROOT/replicas-${SLURM_JOB_ID}" 2000
```

The output directory must be new. Do not treat eight independent process
times as one elapsed time. CUDA MPS is an optional third configuration;
enabling it changes GPU process scheduling but does not turn LAMMPS into an
ALCHEMI batch.

The [reference](../reference/limits.md) distinguishes historical process
measurements from runs qualified with this branch's CLI input. Only compare
numbers measured with the same model, atom count, steps, warm-up convention,
and GPU allocation.
