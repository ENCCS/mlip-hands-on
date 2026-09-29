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
: "${MLIP_ORB_CHECKPOINT:?set the Orb-v3 conservative-inf-omat checkpoint outside Git}"
: "${MLIP_RESULTS_DIR:?set a private, existing results directory}"
command -v pixi jq >/dev/null
manifest="$MLIP_LESSON_ROOT/examples/torchsim/pixi.toml"
test -f "$manifest"
test -f "$MLIP_TORCHSIM_CHECKPOINT"
test -f "$MLIP_ORB_CHECKPOINT"
sha256sum -c - >/dev/null <<SUMS
7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0  $MLIP_TORCHSIM_CHECKPOINT
0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0  $MLIP_ORB_CHECKPOINT
SUMS
test -d "$MLIP_RESULTS_DIR"
export PYTHONNOUSERSITE=1
outdir="$MLIP_RESULTS_DIR/leonardo-orb-${SLURM_JOB_ID}"
test ! -e "$outdir"
mkdir "$outdir"
cd "$MLIP_LESSON_ROOT/examples"
run() {
  srun --nodes=1 --ntasks=1 --gpus=1 --cpu-bind=cores \
    pixi run --frozen --manifest-path "$manifest" python -m "$@"
}
run torchsim.layered --device cuda --dtype float64 \
  --mace-checkpoint "$MLIP_TORCHSIM_CHECKPOINT" \
  --orb-checkpoint "$MLIP_ORB_CHECKPOINT" --outdir "$outdir/graphite"
run torchsim --model orb-v3-conservative-inf-omat --checkpoint "$MLIP_ORB_CHECKPOINT" \
  --device cuda --dtype float64 --n-variants 16 --baseline-n 8 --outdir "$outdir/screening"
jq -e '(.results | length) == 6 and all(.results[]; .converged)' \
  "$outdir/graphite/graphite.json" >/dev/null
jq -e '.n_structures == 64 and .baseline_n == 8 and .batched_wall_s > 0' \
  "$outdir/screening/summary.json" >/dev/null
printf 'Leonardo Orb graphite and screening runs passed: %s\n' "$outdir"
