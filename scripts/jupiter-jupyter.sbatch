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
: "${MLIP_JUPITER_ROOT:?set the private artifact root}"
: "${MLIP_LESSON_ROOT:?set the staged lesson root}"
: "${MLIP_JUPITER_ALCHEMI_ENV_ID:?set the qualified native environment job ID}"
: "${MLIP_JUPITER_JUPYTER_ENV:?set the reviewed private Jupyter environment}"
export MLIP_JOB_LOG_DIR="${MLIP_JOB_LOG_DIR:-$MLIP_JUPITER_ROOT}"
case "$MLIP_JUPITER_ALCHEMI_ENV_ID" in ''|*[!0-9]*) exit 2 ;; esac
test -f "$MLIP_LESSON_ROOT/scripts/start-jupyter.sh"
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 Python/3.12.3 JupyterLab/4.3.4 >/dev/null
test -x "$MLIP_JUPITER_JUPYTER_ENV/bin/python"
export PATH="$MLIP_JUPITER_JUPYTER_ENV/bin:$PATH"
export MLIP_ALCHEMI_RUNNER=native
export MLIP_NATIVE_ALCHEMI_PYTHON="$MLIP_JUPITER_ROOT/env-alchemi-$MLIP_JUPITER_ALCHEMI_ENV_ID/bin/python"
export MLIP_MODEL="$MLIP_JUPITER_ROOT/inputs/mace-mp-0a-small-2ddb079cee0e131e.model"
export MLIP_JUPYTER_TLS=1
test -x "$MLIP_NATIVE_ALCHEMI_PYTHON" && test -f "$MLIP_MODEL" && test -d "$MLIP_JOB_LOG_DIR"
mode=$(stat -c %a "$MLIP_JOB_LOG_DIR")
owner=$(stat -c %u "$MLIP_JOB_LOG_DIR")
if (( (8#$mode & 077) != 0 )) || [[ "$owner" != "$(id -u)" ]]; then
  echo 'The job-log directory must be owner-only (setgid is allowed).' >&2
  exit 1
fi
python -c 'import jupyterlab, jupytext, matplotlib, jupyterlab_myst' >/dev/null
exec srun --nodes=1 --ntasks=1 --gpus=1 --cpu-bind=cores \
  bash "$MLIP_LESSON_ROOT/scripts/start-jupyter.sh"
