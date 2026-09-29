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

Get the complete lesson repository first. It contains the Markdown
notebooks, scripts, build definitions, dependency locks, and example inputs:

```bash
git clone https://github.com/ENCCS/mlip-hands-on.git
cd mlip-hands-on
cp .env.example .env
```

Edit only `.env` with paths for **your** project and prepared artifacts;
it is ignored by Git. The source checkout does not include MACE weights,
SIF images, LAMMPS binaries, site credentials, or private run outputs.
It does include the small reviewed-results CSV used by the offline chapter.
Keep those outside the checkout and check their identities before a run.
The commands in later pages assume your shell starts in the repository root
(the directory containing `examples/` and `scripts/`). For a Jupyter
session, start the server with that same repository as its file root so its
`.md` pages and included source files are visible.

Part B examples use one pinned MACE checkpoint but separate software environments:
JupyterLab for the notebooks, an ALCHEMI SIF on Arrhenius (or a private native
environment on JUPITER), and native Python/LAMMPS for ML-IAP/Kokkos. Keeping
them separate avoids loading two incompatible CUDA or MPI stacks into one
notebook kernel. Part A uses its own pixi environment and model; see its
page.

The core exercises need one GPU and prepared artifacts. Building a SIF and
LAMMPS is covered in the first episodes, but can be done before the GPU session.

:::{important}
No notebook submits a Slurm job. Obtain an allocation and check the selected
model and runtime artifacts before running GPU cells.
:::

## Choose a site

The tabs below point to different tested routes; they do not make the sites
interchangeable. Selecting a site under **Prepare** also selects it under
**Run**. The complete variable list is in
[Environment variables](../reference/environment.md).

### Prepare

::::{tab-set}
:sync-group: site

:::{tab-item} Arrhenius
:sync: arrhenius
Read the [Arrhenius setup](arrhenius.md) for the ALCHEMI SIF and native
MPI-LAMMPS runtime. This is the complete ALCHEMI-and-LAMMPS notebook route.
:::

:::{tab-item} JUPITER
:sync: jupiter
Read the [JUPITER setup](jupiter.md) for its separate native ALCHEMI,
ML-IAP/Kokkos, and Metatomic environments. Do not reuse the Arrhenius SIF
or MPI binary.
:::

:::{tab-item} Leonardo
:sync: leonardo
Read the [Leonardo setup](leonardo.md) for its bounded A100 single-GPU route.
It does not qualify the complete LAMMPS notebook sequence.
:::

::::

### Run

::::{tab-set}
:sync-group: site

:::{tab-item} Arrhenius
:sync: arrhenius
After checking the artifacts and obtaining one GPU, run the
[silicon trajectory](../episodes/04-silicon-md.md),
[ALCHEMI batch](../episodes/05-batched-md.md), and
[LAMMPS replicas](../episodes/06-lammps-replicas.md) in order.
:::

:::{tab-item} JUPITER
:sync: jupiter
Use the [JUPITER-specific job and notebook instructions](jupiter.md).
The ALCHEMI batch notebook has passed a short native functional check;
the full LAMMPS notebook route remains Arrhenius-specific.
:::

:::{tab-item} Leonardo
:sync: leonardo
Use only the [tested Leonardo single-GPU commands](leonardo.md) unless a
separate site check qualifies a broader route.
:::

::::

:::{note}
The lesson uses NVIDIA ALCHEMI **Toolkit**, not the separately packaged
ALCHEMI NIM service.
:::
