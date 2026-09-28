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

# Before you start

The examples use one pinned MACE checkpoint but separate software environments:
JupyterLab for the notebooks, an ALCHEMI SIF on Arrhenius (or a private native
environment on JUPITER), and native Python/LAMMPS for ML-IAP/Kokkos. Keeping
them separate avoids loading two incompatible CUDA or MPI stacks into one
notebook kernel.

The core exercises need one GPU and prepared artifacts. Building a SIF and
LAMMPS is covered in the first episodes, but can be done before the GPU session.
No notebook submits a Slurm job. Obtain an allocation before running GPU cells.

For the complete ALCHEMI-and-LAMMPS notebook route, read the
[Arrhenius setup](arrhenius.md), copy `.env.example` to a private `.env`,
and check each selected artifact before starting a GPU session. The
[JUPITER setup](jupiter.md) qualifies short native ML-IAP and Metatomic LAMMPS
runs, the ALCHEMI MyST batch chapter, and a bounded Jupyter server check.
The full LAMMPS notebook route remains Arrhenius-specific. The complete
variable list is in
[Environment variables](../reference/environment.md).

:::{note}
The lesson uses NVIDIA ALCHEMI **Toolkit**, not the separately packaged
ALCHEMI NIM service.
:::
