#!/usr/bin/env bash
# Run on a JUPITER login node, which has network access.
set -euo pipefail
umask 077
: "${MLIP_JUPITER_ROOT:?set a private artifact root outside Git}"
: "${MLIP_LESSON_ROOT:?set the staged lesson root}"
revision=6a3910424d0aeccf27ad1fc233be1933f72631d2
source="$MLIP_JUPITER_ROOT/lammps-metatomic-patched-$revision"
patch="$MLIP_LESSON_ROOT/patches/metatomic-kokkos-stale-override.patch"
printf '%s  %s\n' 5b6a6edf889de2d14820a6a012d2559b0c4de4b3e82974e3b4a3d2c994376a09 "$patch" | sha256sum -c - >/dev/null
test ! -e "$source"
git clone --no-checkout --filter=blob:none \
  https://github.com/metatensor/lammps.git "$source"
git -C "$source" checkout --detach "$revision"
test "$(git -C "$source" rev-parse HEAD)" = "$revision"
git -C "$source" apply --check "$patch"
git -C "$source" apply "$patch"
git -C "$source" diff --check
printf 'Pinned, locally patched Metatomic LAMMPS source: %s\n' "$source"
