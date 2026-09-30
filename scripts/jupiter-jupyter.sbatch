#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Private one-GPU MyST notebook session. Submit with a private jupyter-%j.log.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_LESSON_ROOT:?set the staged lesson root}"
: "${MLIP_NOTEBOOK_VENV:?set the private notebook environment}"
: "${MLIP_ENV_FILE:?set the private artifact environment file}"
: "${MLIP_JOB_LOG_DIR:?set the private job log directory}"
test -f "$MLIP_LESSON_ROOT/scripts/start-jupyter.sh"
test -f "$MLIP_ENV_FILE"
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 Python/3.12.3 JupyterLab/4.3.4 >/dev/null
test -x "$MLIP_NOTEBOOK_VENV/bin/python"
export PATH="$MLIP_NOTEBOOK_VENV/bin:$PATH"
set -a
source "$MLIP_ENV_FILE"
set +a
export MLIP_JUPYTER_TLS=1
test -f "$MLIP_MODEL" && test -f "$MLIP_ALCHEMI_SIF" && test -d "$MLIP_JOB_LOG_DIR"
mode=$(stat -c %a "$MLIP_JOB_LOG_DIR")
owner=$(stat -c %u "$MLIP_JOB_LOG_DIR")
if (( (8#$mode & 077) != 0 )) || [[ "$owner" != "$(id -u)" ]]; then
  echo 'The job-log directory must be owner-only (setgid is allowed).' >&2
  exit 1
fi
python -c 'import jupyterlab, jupytext, matplotlib, jupyterlab_myst' >/dev/null
# Keep the notebook kernel in the allocation's batch shell; its cells may
# launch MPI-enabled native LAMMPS or separate Slurm steps themselves.
exec bash "$MLIP_LESSON_ROOT/scripts/start-jupyter.sh"
