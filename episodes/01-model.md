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

# The MACE potential and the two MD engines

A molecular-dynamics (MD) engine advances atomic positions and velocities
one step at a time. The resulting sequence of atomic states is a
**trajectory**. The engine needs forces at each step. Here a MACE
machine-learned interatomic potential
predicts those forces for diamond-structure silicon. MACE is a family of
models, not a single universal checkpoint; a model suitable for one chemical
domain may not be suitable for another.

This lesson pins the **MACE-MP-0a small** checkpoint. The original file is
loaded by ALCHEMI Toolkit. A separately exported ML-IAP representation of
the same potential is loaded by LAMMPS. Their file hashes differ because
their formats differ; the export still needs its own validation.

:::{dropdown} model.toml
```{literalinclude} ../reference/model.toml
:language: toml
:lines: 1-13
:lineno-match:
```
:::

Open the complete [model identity file](../reference/model.toml) to see the
checkpoint and export hashes before preparing either engine.

Other MACE families include MACE-MPA, MACE-OMAT, and MACE-OFF. They cover
different training data and intended applications. Listing them here does
not mean that our pinned ALCHEMI image or ML-IAP export can run them without
new preparation and tests. See the
[MACE foundation-model documentation](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html)
when choosing a model for a new system.

ALCHEMI Toolkit builds atomic data structures, calls the model, and can
advance many independent systems together in a **batch**: they share one
program and GPU allocation but do not exchange forces with each other.
LAMMPS provides a compiled MD
engine; ML-IAP lets it call the MACE potential, while Kokkos runs supported
work on the GPU. In the first comparison both engines use one silicon system
on one GPU. Later, ALCHEMI batches independent systems and LAMMPS runs
separate client processes.

:::{note}
The model checkpoint is kept outside the SIF so the image can be built,
inspected, and reused without embedding a large model file or assuming its
redistribution rights. The model is selected explicitly and mounted
read-only at run time. Changing it requires a new identity and a new ML-IAP
export, not just a filename edit.
:::
