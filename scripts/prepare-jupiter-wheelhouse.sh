#!/usr/bin/env bash
# Run on a JUPITER login node: download pinned aarch64 wheels, no installation.
set -euo pipefail
umask 077
: "${MLIP_LESSON_ROOT:?set the staged lesson source root}"
: "${MLIP_WHEELHOUSE_OUTPUT:?set a fresh private .tar path outside Git}"
case "$MLIP_WHEELHOUSE_OUTPUT" in /*.tar) ;; *) exit 2 ;; esac
test ! -e "$MLIP_WHEELHOUSE_OUTPUT"
test "$(uname -m)" = aarch64
build_lock="$MLIP_LESSON_ROOT/locks/build-requirements.lock"
runtime_lock="$MLIP_LESSON_ROOT/locks/requirements.lock"
printf '%s  %s\n' a05f79e6d6005b60de637a829eaa9032e80031d65292126763ee7395f591e514 "$build_lock" | sha256sum -c - >/dev/null
printf '%s  %s\n' e0b85785c55631de2b6a134eabca85b5c4537ed0130805026d96a9ffb895576f "$runtime_lock" | sha256sum -c - >/dev/null
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 Python/3.12.3 >/dev/null
scratch=$(mktemp -d /tmp/mlip-jupiter-wheelhouse.XXXXXX)
mkdir -m 700 "$scratch/wheels"
echo "Private login-node download staging: $scratch"
python3 -m pip download --no-deps --no-build-isolation --require-hashes \
  -r "$build_lock" -d "$scratch/wheels" >"$scratch/build-download.log" 2>&1
python3 -m pip download --no-deps --no-build-isolation --require-hashes \
  -r "$runtime_lock" -d "$scratch/wheels" >"$scratch/runtime-download.log" 2>&1
staged=$(mktemp "${MLIP_WHEELHOUSE_OUTPUT}.partial.XXXXXX")
tar -C "$scratch" -cf "$staged" wheels
sha256sum "$staged"
ln -- "$staged" "$MLIP_WHEELHOUSE_OUTPUT"
rm -- "$staged"
sha256sum "$MLIP_WHEELHOUSE_OUTPUT"
echo 'Pinned wheelhouse archive ready; review the archive before offline install.'
