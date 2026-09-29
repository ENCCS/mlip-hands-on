#!/bin/bash
#SBATCH --job-name=mlip-matgl
#SBATCH --account=<PROJECT>
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=matgl-%j.log
# Run all MatGL tutorial steps on one MI250X GCD. Prepare first on a login node:
#   cd <SCRATCH>/mlip-matgl && module purge && module use /appl/local/csc/modulefiles/ && module load pytorch
#   python -m venv --system-site-packages .venv && source .venv/bin/activate
#   pip install -r examples/matgl/requirements.txt
#   HF_HOME=$PWD/hf-home python examples/matgl/cli.py prefetch
# Submit from <SCRATCH>/mlip-matgl with: sbatch scripts/lumi-matgl.sbatch
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
# float32 det/prod kernels fail on this ROCm stack (error 209); float64 works.
export MATGL_FLOAT_BITS=64
export MPLCONFIGDIR="/tmp/$USER-mpl-$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MPLCONFIGDIR"
trap 'rm -rf "$MIOPEN_USER_DB_PATH" "$MPLCONFIGDIR"' EXIT

cd "$root/examples/matgl"
failed=()
for step in versions relax-md lattice predict train figures; do
    echo "== $step $(date +%T)"
    srun --ntasks=1 --gpus=1 --cpu-bind=cores python cli.py "$step" || failed+=("$step")
done
echo "== done $(date +%T), failed: ${failed[*]:-none}"
