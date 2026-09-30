#!/bin/bash
#SBATCH --job-name=mlip-neb
#SBATCH --account=<PROJECT>
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=neb-%j.log
# CI-NEB for a Li hop in LiFePO4 with three MLIPs on one MI250X GCD. Prepare first on a login node:
#   cd <SCRATCH>/mlip-neb && module purge && module use /appl/local/csc/modulefiles/ && module load pytorch
#   python -m venv --system-site-packages .venv && source .venv/bin/activate
#   pip install -r examples/neb/requirements.txt
#   for m in MACE-MP-0b Orb-v3 TensorNet; do HF_HOME=$PWD/hf-home python examples/neb/cli.py prefetch --model $m; done
# Submit from <SCRATCH>/mlip-neb with: sbatch scripts/lumi-neb.sbatch
set -uo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}"
root="$SLURM_SUBMIT_DIR"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch
source "$root/.venv/bin/activate"

export PYTHONNOUSERSITE=1
export HF_HOME="$root/hf-home" HF_HUB_OFFLINE=1
export MIOPEN_USER_DB_PATH="/tmp/$USER-miopen-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export HIP_FORCE_DEV_KERNARG=1
export MPLCONFIGDIR="/tmp/$USER-mpl-$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MPLCONFIGDIR"
trap 'rm -rf "$MIOPEN_USER_DB_PATH" "$MPLCONFIGDIR"' EXIT

cd "$root/examples/neb"
run() { srun --ntasks=1 --gpus=1 --cpu-bind=cores python cli.py "$@"; }
failed=()
run versions || failed+=(versions)
for model in MACE-MP-0b Orb-v3 TensorNet; do
    echo "== $model $(date +%T)"
    run run --model "$model" || failed+=("$model")
done
run figures || failed+=(figures)
echo "== done $(date +%T), failed: ${failed[*]:-none}"
