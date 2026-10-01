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

# Relax four silicon structures

The four small [starting structures](../../examples/starts/si-relax-01.data) have the same
silicon cell with slightly displaced atoms. The ALCHEMI
[program](../../examples/alchemi_relax.py) batches them and uses FIRE to
relax their positions. LAMMPS reads the same `.data` files and minimizes
them one at a time with Kokkos conjugate gradient in its
[input](../../examples/lammps_relax.in). The algorithms are different.

On an allocated GPU, run the batched ALCHEMI example:

```{literalinclude} ../../examples/alchemi_relax.py
:language: python
:linenos:
```

The Python source constructs four starting systems, batches them, and applies
FIRE. The cell runs that source in the selected SIF:

```{code-cell} ipython3
%%bash
cd ../..
apptainer exec --cleanenv --nv \
  --env TORCH_COMPILE_DISABLE=1 --env TORCH_DISABLE_NATIVE_JIT=1 \
  --bind "$MLIP_MODEL:/models/mace.model:ro" \
  --bind "$PWD/examples/alchemi_relax.py:/opt/mlip/relax.py:ro" \
  --bind "$PWD/examples/starts:/opt/mlip/starts:ro" \
  "$MLIP_ALCHEMI_SIF" python /opt/mlip/relax.py
```

LAMMPS uses a different minimizer, shown in its input. `-var start` selects
one of the same four starting structures:

```{literalinclude} ../../examples/lammps_relax.in
:language: text
:linenos:
```

Run a LAMMPS minimization of the first structure:

On Arrhenius this notebook cell uses a fresh PMI-2 Slurm step for the
native MPICH executable; on JUPITER it runs LAMMPS directly.

```{code-cell} ipython3
%%bash
cd ../..
source "scripts/${MLIP_SITE}-lammps-env.sh"
launch=()
if [[ "$MLIP_SITE" == arrhenius ]]; then
  launch=(srun --mpi=pmi2 --nodes=1 --ntasks=1 --gpus=1)
fi
"${launch[@]}" "$MLIP_LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -log none -in examples/lammps_relax.in \
  -var model "$MLIP_MLIAP_MODEL" \
  -var start "$PWD/examples/starts/si-relax-01.data"
```

Read the reported energies and maximum force to see how relaxation
progressed. The algorithms use different stopping rules, so identical
final coordinates are not expected just because they start from the same
structure and model.

```{note}
FIRE and conjugate gradient are different minimizers. Compare their
reported energies and forces, but do not treat this short example as proof
that they find the same relaxed state.
```
