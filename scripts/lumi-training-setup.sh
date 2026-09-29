#!/bin/bash
# LUMI login node: build a venv on top of the CSC ROCm PyTorch module and prefetch data and model.
# Usage: MLIP_LESSON_ROOT=<SCRATCH>/mlip-hands-on MLIP_TRAINING_DIR=<SCRATCH>/mlip-training \
#          bash scripts/lumi-training-setup.sh
set -euo pipefail
: "${MLIP_LESSON_ROOT:?path to the lesson checkout}"
: "${MLIP_TRAINING_DIR:?working directory on scratch, e.g. <SCRATCH>/mlip-training}"
module purge
module use /appl/local/csc/modulefiles
module load pytorch
export PYTHONNOUSERSITE=1
export HF_HOME="$MLIP_TRAINING_DIR/hf-home"
export PIP_CACHE_DIR="$MLIP_TRAINING_DIR/pip-cache"
mkdir -p "$MLIP_TRAINING_DIR"
venv="$MLIP_TRAINING_DIR/venv"
test -d "$venv" || python -m venv --system-site-packages "$venv"
source "$venv/bin/activate"
pip install --quiet -r "$MLIP_LESSON_ROOT/examples/training/requirements.txt"
python -c "import torch, matgl; print('torch', torch.__version__, 'hip', torch.version.hip, 'matgl', matgl.__version__)"
cd "$MLIP_LESSON_ROOT/examples"
python -m training prefetch --data "$MLIP_TRAINING_DIR/data"
