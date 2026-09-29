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

```{code-cell} ipython3
%%bash
cd ../..
bash scripts/run-alchemi-relax.sh "$PWD" "$MLIP_ALCHEMI_SIF" "$MLIP_MODEL"
```

Then run a LAMMPS minimization of the first structure:

```{code-cell} ipython3
%%bash
cd ../..
bash scripts/run-lammps-relax.sh "$PWD" "$MLIP_LMP" "$MLIP_MLIAP_MODEL" \
  "$PWD/examples/starts/si-relax-01.data"
```

The code prints energies and a maximum force. A smaller final force suggests
relaxation progressed; identical final coordinates are not guaranteed by
different minimizers and numerical implementations. This exercise checks
that both routes run from the same starts, not scientific equivalence.
