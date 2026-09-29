# Before you start

Clone this repository. The lesson source is in `content/`; readable programs
and LAMMPS inputs are in `examples/`; the short commands you run are in
`scripts/`. Nothing in `maintainer/` is required for a participant run.

You need an allocated NVIDIA GPU, the pinned original MACE checkpoint, an
ALCHEMI image, and an MPI-enabled ML-IAP/Kokkos LAMMPS executable. The
checkpoint and built artifacts stay outside Git. The site pages show how to
prepare them.

Copy `.env.example` to a private `.env` and edit the paths. Load it in the
shell with `set -a; . ./.env; set +a`. Start JupyterLab after that so its
notebook cells inherit the same inputs.
[Inputs and paths](../reference/inputs.md) explains the variables.
