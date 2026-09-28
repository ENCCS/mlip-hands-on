#!/usr/bin/env bash
# Run on a JUPITER login node; install only into a fresh private environment.
set -euo pipefail
umask 077
: "${MLIP_JUPITER_ROOT:?set the private artifact root}"
: "${MLIP_JUPITER_MYST_WHEEL:?set the reviewed ENCCS MyST wheel path}"
: "${MLIP_JUPITER_MYST_WHEEL_SHA256:?set its reviewed SHA-256}"
: "${MLIP_JUPITER_JUPYTER_ENV:?set a fresh private environment path}"
[[ "$MLIP_JUPITER_MYST_WHEEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
test -f "$MLIP_JUPITER_MYST_WHEEL"
test ! -e "$MLIP_JUPITER_JUPYTER_ENV"
printf '%s  %s\n' "$MLIP_JUPITER_MYST_WHEEL_SHA256" \
  "$MLIP_JUPITER_MYST_WHEEL" | sha256sum -c - >/dev/null
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 Python/3.12.3 JupyterLab/4.3.4 >/dev/null
python -m venv --system-site-packages "$MLIP_JUPITER_JUPYTER_ENV"
"$MLIP_JUPITER_JUPYTER_ENV/bin/python" -m pip install --no-cache-dir \
  'jupytext==1.17.3' 'matplotlib==3.10.9' \
  >"$MLIP_JUPITER_JUPYTER_ENV/install.log" 2>&1
"$MLIP_JUPITER_JUPYTER_ENV/bin/python" -m pip install --no-deps \
  "$MLIP_JUPITER_MYST_WHEEL" >>"$MLIP_JUPITER_JUPYTER_ENV/install.log" 2>&1
"$MLIP_JUPITER_JUPYTER_ENV/bin/python" - <<'PY'
import jupyterlab, jupytext, matplotlib, jupyterlab_myst
print('Private JUPITER notebook environment is import-ready')
PY
