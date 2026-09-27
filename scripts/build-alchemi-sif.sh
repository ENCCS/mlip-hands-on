#!/bin/bash
set -euo pipefail
# Run on a reviewed aarch64 Apptainer builder, from this repository checkout.
: "${MLIP_SIF_OUTPUT:?set a fresh absolute output .sif path outside Git}"
case "$MLIP_SIF_OUTPUT" in /*.sif) ;; *) echo 'output must be an absolute .sif path' >&2; exit 2;; esac
test "$(uname -m)" = aarch64
test ! -e "$MLIP_SIF_OUTPUT"
command -v apptainer >/dev/null
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test -f "$here/alchemi-aarch64.def"
test -f "$here/locks/requirements.lock"
test -f "$here/locks/build-requirements.lock"
staged="$MLIP_SIF_OUTPUT.partial.$$.sif"
test ! -e "$staged"
trap 'echo "build left staging at: $staged" >&2' ERR
cd "$here"
apptainer build "$staged" alchemi-aarch64.def
apptainer sif list "$staged" >/dev/null
mv -n -- "$staged" "$MLIP_SIF_OUTPUT"
test ! -e "$staged" || { echo 'destination appeared during build; staged SIF retained' >&2; exit 1; }
trap - ERR
sha256sum "$MLIP_SIF_OUTPUT"
