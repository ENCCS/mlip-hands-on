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

# Build native LAMMPS with ML-IAP, Kokkos, and MPI

The native path uses a separate CPython 3.12 environment with Torch, MACE,
and CuPy. ML-IAP embeds Python, so the build also needs matching `Python.h`
and `libpython3.12.so`. The notebook venv is not a substitute for that
development installation.

The supplied Arrhenius build job takes a pinned LAMMPS source archive and a
prepared native Python archive. It checks their identities, then loads the
current reviewed GCC/CUDA environment:

```{literalinclude} ../scripts/build-lammps-mpi.sbatch
:language: bash
:start-at: source /software/sse2/init/hpc_init_sse.sh
:end-at: test -x "$(command -v make)"
```

The CMake configuration enables MPI, ML-IAP, Kokkos, Python, Hopper GPU
code, and Grace CPU code. `mpicc` and `mpicxx` must come from the same site
MPI stack used when running the executable.

```{literalinclude} ../scripts/build-lammps-mpi.sbatch
:language: bash
:start-at: cmake -S
:end-at: -D Python_LIBRARY=
```

The script runs in a separately reviewed Slurm build allocation and writes
an MPI runtime candidate archive. It does not overwrite an existing LAMMPS
installation, submit itself, or prove that multi-GPU MD is scientifically
correct. A one-rank smoke precedes the optional 1/2/4-GPU scaling episode.

## Export the same checkpoint for ML-IAP

The ALCHEMI run reads the original MACE file; LAMMPS reads an ML-IAP export.
After selecting the native Python environment with MACE installed, use a
fresh output path outside Git. The small exporter checks the original model
hash, requires one visible GPU, and refuses to replace an existing export:

```bash
export MLIP_MLIAP_MODEL=/path/outside/git/mace-mp-0a-small-mliap.pt
python examples/export_mace_mliap.py \
  --model "$MLIP_MODEL" --output "$MLIP_MLIAP_MODEL"
```

Record the printed export hash with the runtime identity. For the pinned
Arrhenius build it should match the value in `reference/model.toml`. A fresh
export did match that value and completed a short one-rank LAMMPS run. On
another site, check the export and one-rank run again; a matching hash alone
is not a force or trajectory validation.

On Arrhenius, the tested multi-rank runner uses site MPICH with PMI2/CXI,
`gpu/aware on`, and peer-visible GPUs. Hiding every other GPU from a rank
caused an earlier ML-IAP ghost-exchange failure. These are implementation
details of this site profile, not portable defaults. On another cluster,
recheck the compiler/MPI ABI, device assignment, network transport, and
Slurm options. A future Leonardo profile should live alongside this one,
not be mixed into the Arrhenius commands.

NCCL matters to ALCHEMI's separate distributed `DomainParallel` path; it
is not a replacement for the MPI configuration of this LAMMPS build.
