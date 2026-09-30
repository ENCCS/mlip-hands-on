# Before you start

On an HPC login node, clone the current lesson branch into storage that is
also visible from its GPU nodes:

```bash
git clone --branch lesson/minimal-md-cli --single-branch \
  https://github.com/ENCCS/mlip-hands-on.git
cd mlip-hands-on
export MLIP_LESSON_ROOT="$PWD"
```

The lesson source is in `content/`; readable programs and LAMMPS inputs are
in `examples/`; the short commands you run are in `scripts/`. Nothing in
`maintainer/` is required for a participant run.

Create a private environment file outside the checkout. Replace the example
directory below with your own project-storage path before running the block.
It must be accessible from the GPU node. The `test` prevents overwriting an
existing file:

```bash
export MLIP_PRIVATE_DIR=/path/to/your/private/project-storage
mkdir -p -m 700 "$MLIP_PRIVATE_DIR"
export MLIP_ENV_FILE="$MLIP_PRIVATE_DIR/mlip-lesson.env"
test ! -e "$MLIP_ENV_FILE" && install -m 600 .env.example "$MLIP_ENV_FILE"
${EDITOR:-vi} "$MLIP_ENV_FILE"
```

In that file, choose `MLIP_SITE=arrhenius` or `MLIP_SITE=jupiter`, and replace
the artifact paths with your own. After editing, load the values in the
**same Bash shell** that will run the examples:

```bash
set -a
source "$MLIP_ENV_FILE"
set +a
printf 'Selected site: %s\n' "$MLIP_SITE"
```

You need an allocated NVIDIA GPU, the pinned original MACE checkpoint, an
ALCHEMI image, and an MPI-enabled ML-IAP/Kokkos LAMMPS executable. The
checkpoint and built artifacts stay outside Git. Follow the
[Arrhenius](arrhenius.md) or [JUPITER](jupiter.md) setup page to prepare them
before running an MD example. Do not submit a GPU job just to test the clone
or the environment file.

[Inputs and paths](../reference/inputs.md) explains each variable. To use the
same pages as notebooks, continue with [Open the MyST notebook](notebook.md)
after preparing the site artifacts and Jupyter environment.
