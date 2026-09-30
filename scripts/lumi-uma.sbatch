#!/bin/bash
#SBATCH --job-name=mlip-uma
#SBATCH --account=<PROJECT>
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=uma-%j.log
# UMA 1.2 small (omat and omol tasks) on one MI250X GCD. Prepare first on a login node:
#   cd <SCRATCH>/mlip-uma && module purge && module use /appl/local/csc/modulefiles/ && module load pytorch
#   python -m venv --system-site-packages .venv && source .venv/bin/activate
#   pip install --no-deps fairchem-core==2.21.0 && pip install -r examples/uma/requirements.txt
#   export HF_HOME=$PWD/hf-home FAIRCHEM_CACHE_DIR=$PWD/fairchem-cache HF_TOKEN_PATH=$HOME/.cache/huggingface/token
#   python examples/uma/cli.py prefetch
# Submit from <SCRATCH>/mlip-uma with: sbatch scripts/lumi-uma.sbatch
set -uo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}"
root="$SLURM_SUBMIT_DIR"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch
source "$root/.venv/bin/activate"

export PYTHONNOUSERSITE=1
export HF_HOME="$root/hf-home" HF_HUB_OFFLINE=1 FAIRCHEM_CACHE_DIR="$root/fairchem-cache"
export MIOPEN_USER_DB_PATH="/tmp/$USER-miopen-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export HIP_FORCE_DEV_KERNARG=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT

cd "$root/examples/uma"
run() { srun --ntasks=1 --gpus=1 --cpu-bind=cores python cli.py "$@"; }
failed=()
run versions || failed+=(versions)
echo "== run $(date +%T)"
run run --dtype float64 || failed+=(run)
for dtype in float32 float64; do
    echo "== timing $dtype $(date +%T)"
    run timing --dtype "$dtype" || failed+=("timing-$dtype")
done
echo "== done $(date +%T), failed: ${failed[*]:-none}"
