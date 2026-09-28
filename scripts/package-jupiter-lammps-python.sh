#!/usr/bin/env bash
# Package the Python wrapper for one verified private JUPITER LAMMPS build.
set -euo pipefail
umask 077
: "${MLIP_JUPITER_ROOT:?set a private JUPITER artifact root}"
: "${MLIP_JUPITER_BUILD_ID:?set the successful build job ID}"
case "$MLIP_JUPITER_BUILD_ID" in ''|*[!0-9]*) exit 2 ;; esac
test "$(uname -m)" = aarch64

archive="$MLIP_JUPITER_ROOT/inputs/lammps-stable_22Jul2025_update6-c52ecdb29dcf464a58e9cbf03fddab0c9e5389df31923347f55a4d8ce4dd7dd0.tar.gz"
build="$MLIP_JUPITER_ROOT/builds/lammps-mliap-$MLIP_JUPITER_BUILD_ID"
venv="$MLIP_JUPITER_ROOT/env-mace-0315"
wheel_dir="$build/python-wheel-clean-env"
test -f "$archive" && test -f "$build/lmp.sha256" && test -x "$venv/bin/python"
test -f "$build/prefix/lib64/liblammps.so.0"
printf '%s  %s\n' c52ecdb29dcf464a58e9cbf03fddab0c9e5389df31923347f55a4d8ce4dd7dd0 "$archive" | sha256sum -c - >/dev/null
(cd "$build" && sha256sum -c lmp.sha256 >/dev/null)
test ! -e "$wheel_dir"
mkdir -m 700 "$wheel_dir"

module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 CUDA/12 Python/3.12.3 PyTorch/2.5.1 >/dev/null
# python/examples contains symlinks into the source tree, so extract the
# complete pinned archive rather than a broken partial python subtree.
tar -xzf "$archive" -C "$wheel_dir"
cd "$wheel_dir"
env -u PYTHONPATH -u PYTHONHOME "$venv/bin/python" lammps-lammps-751b42d/python/install.py \
  --package "$wheel_dir/lammps-lammps-751b42d/python/lammps" \
  --lib "$build/prefix/lib64/liblammps.so.0" \
  --versionfile "$wheel_dir/lammps-lammps-751b42d/src/version.h" \
  --wheeldir "$wheel_dir" --noinstall >package.log 2>&1
shopt -s nullglob
wheels=("$wheel_dir"/lammps-*.whl)
test "${#wheels[@]}" = 1
"$venv/bin/python" -m pip install --no-index --no-deps "${wheels[0]}" >install.log 2>&1
"$venv/bin/python" -c 'import lammps; print("LAMMPS Python wrapper imported")'
sha256sum "${wheels[0]}"
