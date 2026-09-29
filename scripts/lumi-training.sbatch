#!/bin/bash
#SBATCH --job-name=mlip-training
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
# Submit with: sbatch --account=<PROJECT> --export=ALL,MLIP_LESSON_ROOT=...,MLIP_TRAINING_DIR=... this file
set -euo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_LESSON_ROOT:?path to the lesson checkout}"
: "${MLIP_TRAINING_DIR:?directory prepared by scripts/lumi-training-setup.sh}"
data="$MLIP_TRAINING_DIR/data"
test -f "$data/mp_eform_subset.json"
test -f "$data/matpes_r2scan_Li.jsonl"

module purge
module use /appl/local/csc/modulefiles
module load pytorch
source "$MLIP_TRAINING_DIR/venv/bin/activate"

# ROCm hygiene: private per-job MIOpen and temporary directories, no user site-packages, no network.
export PYTHONNOUSERSITE=1
export HF_HOME="$MLIP_TRAINING_DIR/hf-home" HF_HUB_OFFLINE=1
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export TMPDIR="/tmp/${USER}-tmp-${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export HIP_FORCE_DEV_KERNARG=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MPLCONFIGDIR"
trap 'rm -rf "$MIOPEN_USER_DB_PATH" "$TMPDIR"' EXIT

outdir="$MLIP_TRAINING_DIR/results-$SLURM_JOB_ID"
cd "$MLIP_LESSON_ROOT/examples"
run() { srun --ntasks=1 --gpus=1 --cpu-bind=cores python "$@"; }
run -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
run -m training eform --data "$data" --outdir "$outdir/eform" --accelerator gpu
run -m training finetune --data "$data" --outdir "$outdir/finetune" --accelerator gpu
printf 'Results: %s\n' "$outdir"
