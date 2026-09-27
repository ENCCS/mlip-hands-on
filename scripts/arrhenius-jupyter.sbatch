#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Submitted by submit-arrhenius-jupyter.sh. Keep its log private: it has a token URL.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?run this as a Slurm job}"
: "${MLIP_LESSON_ROOT:?set the staged lesson root}"
: "${MLIP_NOTEBOOK_VENV:?set the private GPU-node notebook venv}"
: "${MLIP_ENV_FILE:?set the private artifact environment file}"
: "${MLIP_JOB_LOG_DIR:?set the private job log directory}"
test -f "$MLIP_LESSON_ROOT/scripts/start-jupyter.sh"
test -x "$MLIP_NOTEBOOK_VENV/bin/python"
test -f "$MLIP_ENV_FILE"
test -d "$MLIP_JOB_LOG_DIR"

source /software/sse2/init/hpc_init_sse.sh
module purge
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 \
  Core/GCCcore/14.3.0 JupyterLab/4.4.9 \
  Core/GCC/14.3.0 matplotlib/3.10.5

export PATH="$MLIP_NOTEBOOK_VENV/bin:$PATH"
export PYTHONPATH="$MLIP_NOTEBOOK_VENV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export JUPYTERLAB_DIR="$MLIP_NOTEBOOK_VENV/share/jupyter/lab"
export JUPYTER_PATH="$MLIP_NOTEBOOK_VENV/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
set -a
source "$MLIP_ENV_FILE"
set +a
export MLIP_JUPYTER_TLS=1
exec bash "$MLIP_LESSON_ROOT/scripts/start-jupyter.sh"
