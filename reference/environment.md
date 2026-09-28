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

# Environment variables

The repository ships `.env.example`. Copy it to `.env`, edit the paths for
your allocation, and keep that file private. The example uses
`snicXXXX-XX-X` and `USER` as placeholders; neither is a real project or
account. A model file, SIF, native runtime, and results directory are
artifacts outside Git.

| Variable | Example or source | Used for |
| --- | --- | --- |
| `MLIP_PROJECT_ROOT` | `/nobackup/proj/disk/snicXXXX-XX-X/personal/USER/mlip-md` | Convenience base path in `.env.example` only. |
| `MLIP_MODEL` | `$MLIP_PROJECT_ROOT/artifacts/models/mace-mp-0a-small.model` | Original pinned MACE checkpoint mounted into the SIF. |
| `MLIP_MLIAP_MODEL` | `$MLIP_PROJECT_ROOT/artifacts/models/mace-mp-0a-small-mliap.pt` | Validated LAMMPS ML-IAP export of the checkpoint. |
| `MLIP_ALCHEMI_SIF` | `$MLIP_PROJECT_ROOT/artifacts/alchemi-aarch64.sif` | ALCHEMI runtime image selected for MD. |
| `MLIP_NATIVE_PREFIX` | `$MLIP_PROJECT_ROOT/artifacts/lammps-mpi` | Extracted native Python and MPI-LAMMPS runtime root. |
| `MLIP_RESULTS_DIR` | `$MLIP_PROJECT_ROOT/results` | Existing private directory for benchmark output. |
| `MLIP_JUPYTER_PORT` | `8888` | Compute-node loopback port; Jupyter token remains private. |
| `MLIP_SIF_OUTPUT` | A new `.sif` path outside Git | Destination when building a new image. |
| `MLIP_NATIVE_ARCHIVE` | Pinned native Python archive | Build input containing CPython, headers, shared library, and packages. |
| `MLIP_LAMMPS_SOURCE_ARCHIVE` | Pinned LAMMPS source archive | MPI-LAMMPS build input. |
| `MLIP_MPI_BUILD_ROOT` | Existing private artifact directory | Destination for a distinct MPI build candidate. |
| `MLIP_MPI_ARCHIVE` | Verified MPI runtime archive | Input for the 1/2/4-GPU scaling job. |
| `MLIP_MPI_ARCHIVE_SHA256` | 64-character digest | Verifies that archive before extraction. |
| `MLIP_JUPITER_ROOT` | `/e/project1/<PROJECT>/<USER>/mlip-md-lesson` | Private JUPITER inputs, build candidates, and results, outside Git. |
| `MLIP_JUPITER_BUILD_ID` | Successful build job ID | Selects the exact native JUPITER build for Python packaging and runtime checks. |
| `MLIP_WHEELHOUSE_OUTPUT` | Fresh private `.tar` path | Login-node destination for pinned aarch64 Python packages. |
| `MLIP_JUPITER_WHEELHOUSE` | Reviewed wheelhouse tar path | Offline Python package input on a JUPITER compute node. |
| `MLIP_JUPITER_WHEELHOUSE_SHA256` | 64-character digest | Verifies the wheelhouse before installation. |
| `MLIP_JUPITER_ALCHEMI_ENV_ID` | Successful environment job ID | Selects one fresh native ALCHEMI environment. |
| `MLIP_JUPITER_REPLICAS` | `1` or `8` | Selects the short one-GPU ALCHEMI functional smoke. |
| `MLIP_ALCHEMI_RUNNER` | `native` on JUPITER, `container` on Arrhenius | Selects how the same notebook cell starts ALCHEMI. |
| `MLIP_NATIVE_ALCHEMI_PYTHON` | Reviewed private Python executable | Runs the pinned native ALCHEMI example on JUPITER. |
| `MLIP_JUPITER_MYST_WHEEL` / `MLIP_JUPITER_MYST_WHEEL_SHA256` | Reviewed private wheel and digest | Input for the private JUPITER MyST renderer; the selected build is hash-checked. |
| `MLIP_JUPITER_JUPYTER_ENV` | Fresh private environment path | JupyterLab/Jupytext/Matplotlib environment for the JUPITER notebook server. |
| `MLIP_METATOMIC_SOURCE` | Patched private LAMMPS checkout | Pinned Metatomic fork with the reviewed Kokkos compatibility patch. |
| `MLIP_METATOMIC_OVERLAY` | Private package directory | Metatomic build/runtime and export dependencies. |
| `MLIP_MACE_EXPORT_OVERLAY` | Private package directory | MACE 0.3.14 needed by the experimental Metatomic exporter. |
| `MLIP_METATOMIC_EXPORT_DIR` | Fresh private output directory | Receives `model.pt`, export options, and diagnostic log. |
| `MLIP_JUPITER_METATOMIC_BUILD_ID` | Successful build job ID | Selects the Metatomic/Kokkos LAMMPS executable. |
| `MLIP_METATOMIC_MODEL` / `MLIP_METATOMIC_MODEL_SHA256` | Exported file and reviewed digest | Select and verify the Metatomic model before a run. |
| `MLIP_METATOMIC_INPUT` | Staged silicon input | Optional 512-atom multi-GPU input; default is 64 atoms. |
| `MLIP_LESSON_ROOT` | Private source stage on a site | Selects the exact lesson example scripts inside a site job. |
| `MLIP_LOCAL_RANK` | Set by a launcher when needed | Optional rank annotation in the ALCHEMI result. Do not set for a one-GPU notebook. |
| `MLIP_ALLOCATED_CUDA_DEVICES` | Set inside the MPI job | Preserves the allocation's peer-visible GPU list for each rank. Do not set manually. |

The scripts check additional standard site variables such as `SLURM_JOB_ID`,
`SLURM_NTASKS`, and `CUDA_VISIBLE_DEVICES`. Their values come from the
allocation, not `.env`. Recheck the actual site policy and artifact hashes
before a class; this page does not authorize a scheduler submission.
