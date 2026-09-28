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

# JUPITER setup: native LAMMPS first

JUPITER's booster nodes use four GH200 GPUs and aarch64 CPUs. This profile
has passed short one-, two-, four-, and eight-GPU silicon runs with the pinned
MACE ML-IAP export; the eight-GPU run used two nodes. These are functional
checks, **not** throughput or scientific-equivalence benchmarks. The ALCHEMI
environment and a Jupyter session are not yet qualified here. Do not use an
Arrhenius SIF or MPI binary here.

The tested native module family is:

```bash
module purge
module load Stages/2025 GCC/13.3.0 CMake/3.29.3 CUDA/12 \
  OpenMPI/5.0.5 Python/3.12.3 PyTorch/2.5.1
```

Use a private project directory outside the lesson Git repository, for
example `/e/project1/<PROJECT>/<USER>/mlip-md-lesson`. Put the pinned LAMMPS
source archive and model in its `inputs/` directory; check their hashes in
`reference/model.toml` and the build script. Set `MLIP_JUPITER_ROOT` to that
private directory. The candidate builder verifies the source archive, then
builds OpenMPI, CUDA/Hopper Kokkos, ML-IAP, and a shared LAMMPS library:

```bash
export MLIP_JUPITER_ROOT=/e/project1/<PROJECT>/<USER>/mlip-md-lesson
sbatch --account=<PROJECT> scripts/build-jupiter-lammps-mpi.sbatch
```

Do not repeat a submission if its outcome is uncertain. Check the job and
the candidate directory first. A completed build is not yet a run. Package
its Python wrapper from the *same* pinned source and shared library:

```bash
export MLIP_JUPITER_BUILD_ID=<completed-build-job-id>
bash scripts/package-jupiter-lammps-python.sh
```

The wrapper goes into a private Python environment; the script never
installs it into the site modules. That environment needs the MACE checkpoint
dependencies and `cuequivariance` plus `cuequivariance-torch` 0.11.1 for the
validated ML-IAP export. The latter two versions match the tested exporter
environment. A working `import lammps` alone does not establish a GPU run.

JUPITER's **batch shell can see all four GPUs** even when the job requested
one. The tested one-GPU script runs its Python calculation inside
`srun --gpus=1`, where PyTorch sees one device. Do not replace that step with
a direct batch-shell Python call when testing this lesson's one-GPU code.

The pinned original checkpoint did not export locally with the first JUPITER
MACE environment: its conversion failed while loading symmetric-contraction
weights. The one-GPU runtime check instead used an independently hashed
ML-IAP export made from the same checkpoint. Recheck the export recipe before
teaching local export as a completed JUPITER step. The source and runtime
must remain two separate claims.

For a short check of that exact export, put it in the private `inputs/`
directory as `mace-mliap-db578c55.pt`, stage this repository outside the
allocation, and submit the smoke script with both paths in the Slurm
environment:

```bash
export MLIP_LESSON_ROOT=/path/to/staged/mlip-md-lesson
sbatch --account=<PROJECT> --export=ALL,MLIP_JUPITER_ROOT,MLIP_LESSON_ROOT,MLIP_JUPITER_BUILD_ID \
  scripts/test-jupiter-lammps-verified.sbatch
```

The smoke runs 64 silicon atoms for five measured NVE steps. It checks that
LAMMPS, ML-IAP/Kokkos, the wrapper, model and GPU step execute together; it
does not measure throughput or establish scientific agreement. Do not submit
it until the private export has been verified against the hash in the script.

For one coupled 512-atom trajectory, the MPI script uses a distinct Slurm GPU
binding on each rank. JUPITER's documented job examples request both `--gres`
and `--gpus-per-task=1`; each rank then sees its own GPU as device zero.
The shared-visibility mode used on Arrhenius is not selected here. For example,
the one-node four-GPU functional check is:

```bash
sbatch --account=<PROJECT> --nodes=1 --ntasks=4 \
  --gres=gpu:4 --gpus-per-task=1 --cpus-per-task=8 \
  --mem=256G --time=00:15:00 \
  --export=ALL,MLIP_JUPITER_ROOT,MLIP_LESSON_ROOT,MLIP_JUPITER_BUILD_ID \
  scripts/test-jupiter-lammps-mpi.sbatch
```

The same script accepts one node with one or two ranks, or two nodes with
eight ranks (four per node). Change the `--nodes`, `--ntasks`, `--gres`, and
`--ntasks-per-node` requests together; `--gres=gpu:4` is **per node** for the
two-node case. It runs ten measured NVE steps, writes one private result per
rank, and verifies completion. On this system a partial-GPU request can still
allocate an exclusive whole node, so do not read the requested GPU count as
the billed allocation. Inspect current policy and queue before submitting.

The tested one-node two- and four-GPU and two-node eight-GPU runs completed.
At 512 atoms and ten measured steps, rank-wise final energies were internally
identical within each job but differed slightly across rank counts. These
checks do not establish decomposition-independent scientific agreement or
useful scaling.

The `container`-group route is intentionally omitted from this profile.
Native LAMMPS does not require that group. A future ALCHEMI virtual
environment needs separate qualification.
