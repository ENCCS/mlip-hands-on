# Arrhenius setup

These commands are an Arrhenius example, not a portable Slurm policy. Check
the current account, reservation, partition, modules, and GPU allocation
rules before using them. Substitute your own project number and user name;
do not put them in the lesson repository.

The documentation environment is separate from the GPU notebook environment.
Build the Sphinx pages on a login or local development machine with
`requirements.txt`; do not assume that environment works on the aarch64 GPU
node. The tested notebook modules are loaded **inside** an allocation:

```bash
source /software/sse2/init/hpc_init_sse.sh
module purge
module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 \
  Core/GCCcore/14.3.0 JupyterLab/4.4.9 \
  Core/GCC/14.3.0 matplotlib/3.10.5
```

The site's JupyterLab 4.4.9 is older than the patched renderer's JupyterLab
4.6.4 bundle. Do **not** install the renderer into the site application.
Instead, make a private Python 3.13 environment outside Git on a GH200 node.
The repository file `jupyterlab-enccs/README.md` specifies how to build the
pinned renderer wheel on a development machine. With the modules above loaded:

```bash
export MLIP_NOTEBOOK_VENV=/path/outside/git/mlip-notebook-venv
python3 -m venv --system-site-packages "$MLIP_NOTEBOOK_VENV"
"$MLIP_NOTEBOOK_VENV/bin/python" -m pip install \
  jupyterlab==4.6.4 jupyterlab-server==2.28.1 jupyter-server==2.21.1 \
  jupytext==1.19.5 /path/to/private/jupyterlab_myst-2.4.2-py3-none-any.whl
```

On later allocations, load the same modules, select the private app and
Python packages **ahead of** the module-provided ones, then start Jupyter:

```bash
export MLIP_NOTEBOOK_VENV=/path/outside/git/mlip-notebook-venv
export PATH="$MLIP_NOTEBOOK_VENV/bin:$PATH"
export PYTHONPATH="$MLIP_NOTEBOOK_VENV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export JUPYTERLAB_DIR="$MLIP_NOTEBOOK_VENV/share/jupyter/lab"
export JUPYTER_PATH="$MLIP_NOTEBOOK_VENV/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
set -a; source .env; set +a
bash scripts/start-jupyter.sh
```

Prepare the private environment once for the aarch64 node, not on the x86
login node. Keep both it and `.env` outside Git. Verify `jupyter
labextension list` shows both `jupyterlab-myst` and `jupyterlab-jupytext` as
`OK`. The site modules set their own Python and Jupyter paths; removing those
paths entirely also removes needed packages, so prepend the private paths as
shown rather than clearing them. The tested server accepted an authenticated
request for a lesson `.md` file and returned a notebook with code cells.

The notebook server binds to compute-node loopback and keeps token
authentication. It does not allocate a GPU or open a laptop tunnel. See
[Opening the notebook](notebook.md) for the connection concept.

For the native MPI build, the tested Arrhenius module family is loaded by
`scripts/build-lammps-mpi.sbatch`:

```bash
source /software/sse2/init/hpc_init_sse.sh
module load GPU/buildenv-gcccuda/2026.03-cu13.0
```

The MPI candidate uses the site MPICH wrappers and a CPython 3.12 root that
contains `Python.h` and `libpython3.12.so`. The notebook packages above are
not that build dependency. The run script uses PMI2/CXI and explicit peer GPU
visibility; these are site-specific, not universal LAMMPS settings. Recheck
them after toolchain or Slurm changes.
