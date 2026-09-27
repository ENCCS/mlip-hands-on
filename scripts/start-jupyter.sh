#!/bin/bash
# Run *inside* one reviewed GPU allocation; creates no Slurm job or tunnel.
set -euo pipefail
: "${SLURM_JOB_ID:?start this only inside an allocated GH200 shell}"
command -v jupyter >/dev/null || { echo 'Jupyter is not installed in this environment' >&2; exit 1; }
python -c 'import jupytext, matplotlib; from jupytext import TextFileContentsManager' >/dev/null || {
  echo 'Jupytext and Matplotlib are required for all MyST notebook chapters' >&2
  exit 1
}
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
settings_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlip-jupyter-settings.XXXXXXXX")
install -D -m 0600 \
  "$here/jupyterlab-settings/@jupyterlab/docmanager-extension/plugin.jupyterlab-settings" \
  "$settings_dir/@jupyterlab/docmanager-extension/plugin.jupyterlab-settings"
export JUPYTERLAB_SETTINGS_DIR="$settings_dir"
cd "$here"
exec jupyter lab --no-browser --ip=127.0.0.1 --port="${MLIP_JUPYTER_PORT:-8888}" \
  --ServerApp.allow_remote_access=False --ServerApp.root_dir="$here" \
  --ServerApp.contents_manager_class=jupytext.TextFileContentsManager
