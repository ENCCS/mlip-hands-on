#!/usr/bin/env bash
# Execute the published MyST page inside one separately approved GPU job.
# This script does not submit a job or publish a result.
set -euo pipefail
: "${SLURM_JOB_ID:?run only in a Slurm job}"
: "${MLIP_LESSON_ROOT:?set the exact staged lesson root}"
: "${MLIP_ENV_FILE:?set the private site env file}"
: "${MLIP_NOTEBOOK_VENV:?set the reviewed notebook environment}"
test -f "$MLIP_ENV_FILE"
test -f "$MLIP_LESSON_ROOT/maintainer/qualify-notebook.py"
test -x "$MLIP_NOTEBOOK_VENV/bin/python"

set -a
source "$MLIP_ENV_FILE"
set +a
case "$MLIP_SITE" in
  arrhenius)
    source /software/sse2/init/hpc_init_sse.sh >/dev/null 2>&1
    module purge >/dev/null 2>&1
    module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 \
      Core/GCCcore/14.3.0 JupyterLab/4.4.9 \
      Core/GCC/14.3.0 matplotlib/3.10.5 >/dev/null 2>&1
    : "${MLIP_NATIVE_RUNTIME_ARCHIVE:?Arrhenius native Python archive missing}"
    test -f "$MLIP_NATIVE_RUNTIME_ARCHIVE"
    scratch=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
    test -d "$scratch" && test -w "$scratch"
    runtime=$(mktemp -d "$scratch/mlip-notebook-native.XXXXXXXX")
    cleanup_runtime() {
      if [[ "$runtime" == "$scratch"/mlip-notebook-native.* && -d "$runtime" ]]; then
        rm -rf -- "$runtime"
      fi
    }
    trap cleanup_runtime EXIT
    tar -xzf "$MLIP_NATIVE_RUNTIME_ARCHIVE" -C "$runtime"
    export MLIP_NATIVE_PYTHON="$runtime/python"
    export PYTHONPATH="$MLIP_NOTEBOOK_VENV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    export JUPYTERLAB_DIR="$MLIP_NOTEBOOK_VENV/share/jupyter/lab"
    export JUPYTER_PATH="$MLIP_NOTEBOOK_VENV/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
    # The notebook kernel runs in the batch shell. Its LAMMPS cell creates
    # a fresh Slurm PMI-2 step; %%bash closes inherited PMI file descriptors.
    notebook_step=()
    ;;
  jupiter)
    module purge >/dev/null 2>&1
    module load Stages/2025 GCC/13.3.0 Python/3.12.3 JupyterLab/4.3.4 >/dev/null 2>&1
    : "${MLIP_NATIVE_PYTHON:?matching native Python missing}"
    # Keep the notebook kernel in the batch shell. A notebook %%bash cell
    # launches native LAMMPS, whose MPI singleton must not inherit an srun
    # step's PMI descriptors.
    notebook_step=()
    ;;
  *) echo 'unsupported MLIP_SITE' >&2; exit 2 ;;
esac
export PATH="$MLIP_NOTEBOOK_VENV/bin:$PATH"
test -x "$MLIP_NATIVE_PYTHON/bin/python3" || test -x "$MLIP_NATIVE_PYTHON/bin/python"
"${notebook_step[@]}" "$MLIP_NOTEBOOK_VENV/bin/python" \
  "$MLIP_LESSON_ROOT/maintainer/qualify-notebook.py"
