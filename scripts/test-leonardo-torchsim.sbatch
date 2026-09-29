#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_LESSON_ROOT:?stage the lesson outside the allocation}"
: "${MLIP_TORCHSIM_CHECKPOINT:?set the MACE-MP-0b small checkpoint outside Git}"
: "${MLIP_RESULTS_DIR:?set a private, existing results directory}"
command -v pixi >/dev/null
manifest="$MLIP_LESSON_ROOT/examples/torchsim/pixi.toml"
test -f "$manifest"
test -f "$MLIP_TORCHSIM_CHECKPOINT"
test -d "$MLIP_RESULTS_DIR"
export PYTHONNOUSERSITE=1
outdir="$MLIP_RESULTS_DIR/leonardo-torchsim-${SLURM_JOB_ID}"
test ! -e "$outdir"
cd "$MLIP_LESSON_ROOT/examples"
srun --nodes=1 --ntasks=1 --gpus=1 --cpu-bind=cores \
  pixi run --frozen --manifest-path "$manifest" \
  python -m torchsim --model mace-small --checkpoint "$MLIP_TORCHSIM_CHECKPOINT" \
  --device cuda --dtype float64 --n-variants 16 --baseline-n 8 --outdir "$outdir"
jq -e '.n_structures == 64 and .baseline_n == 8 and .batched_wall_s > 0' \
  "$outdir/summary.json" >/dev/null
printf 'Leonardo TorchSim batched relaxation passed: %s\n' "$outdir"
