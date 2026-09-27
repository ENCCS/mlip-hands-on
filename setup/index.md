# Before you start

The examples use one pinned MACE checkpoint but three software environments:
JupyterLab for the notebooks, an ALCHEMI SIF for its Python/CUDA stack, and a
native Python/LAMMPS installation for ML-IAP/Kokkos. Keeping them separate
avoids loading two incompatible CUDA or MPI stacks into one notebook kernel.

The core exercises need one GPU and prepared artifacts. Building a SIF and
LAMMPS is covered in the first episodes, but need not happen during a class.
No notebook submits a Slurm job. Obtain an allocation before running GPU cells.

Read the [Arrhenius setup](arrhenius.md), copy `.env.example` to a private
`.env`, and check each selected artifact before starting a GPU session. The
complete variable list is in [Environment variables](../reference/environment.md).

:::{note}
The lesson uses NVIDIA ALCHEMI **Toolkit**, not the separately packaged
ALCHEMI NIM service.
:::
