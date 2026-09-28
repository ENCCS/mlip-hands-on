#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# Build one private native ALCHEMI candidate; do not replace a working venv.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_JUPITER_ROOT:?set a private artifact root}"
: "${MLIP_LESSON_ROOT:?set the staged lesson source root}"
: "${MLIP_JUPITER_WHEELHOUSE:?set the reviewed private wheelhouse tar}"
: "${MLIP_JUPITER_WHEELHOUSE_SHA256:?set its reviewed SHA-256}"
test "$(uname -m)" = aarch64
build_lock="$MLIP_LESSON_ROOT/locks/build-requirements.lock"
runtime_lock="$MLIP_LESSON_ROOT/locks/requirements.lock"
printf '%s  %s\n' a05f79e6d6005b60de637a829eaa9032e80031d65292126763ee7395f591e514 "$build_lock" | sha256sum -c - >/dev/null
printf '%s  %s\n' e0b85785c55631de2b6a134eabca85b5c4537ed0130805026d96a9ffb895576f "$runtime_lock" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$MLIP_JUPITER_WHEELHOUSE_SHA256" "$MLIP_JUPITER_WHEELHOUSE" | sha256sum -c - >/dev/null
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 Python/3.12.3 >/dev/null
scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mlip-jupiter-wheels-${SLURM_JOB_ID}.XXXXXX")
tar -C "$scratch" -xf "$MLIP_JUPITER_WHEELHOUSE"
candidate="$MLIP_JUPITER_ROOT/env-alchemi-$SLURM_JOB_ID"
test ! -e "$candidate"
python3 -m venv "$candidate"
"$candidate/bin/python" -m pip install --no-index --find-links="$scratch/wheels" \
  --no-cache-dir --no-deps --require-hashes \
  -r "$build_lock" >"$candidate/build-install.log" 2>&1
"$candidate/bin/python" -m pip install --no-index --find-links="$scratch/wheels" \
  --no-cache-dir --no-deps --no-build-isolation \
  --require-hashes -r "$runtime_lock" >"$candidate/runtime-install.log" 2>&1
"$candidate/bin/python" - <<'PY'
import platform
import torch
import nvalchemi
import mace
assert platform.machine() == 'aarch64'
assert torch.version.cuda == '12.6'
assert torch.cuda.is_available()
print('JUPITER native ALCHEMI candidate imports and sees CUDA')
PY
