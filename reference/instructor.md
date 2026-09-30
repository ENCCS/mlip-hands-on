---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Instructor guide

Prepare the model, ML-IAP export, ALCHEMI SIF, native MPI-LAMMPS runtime,
and private results directory before the session. Their source and content
identities must be reviewed together; a filename alone is not enough. Keep
the actual `.env` outside Git. Test the exact artifact set in a short GPU
allocation, then end that allocation.

For a live class, obtain one GH200 allocation under current site rules.
Load and select the private Jupyter environment exactly as in the
[Arrhenius setup](../setup/arrhenius.md), then load `.env` and run
`bash scripts/start-jupyter.sh` from the repository root. It starts only
JupyterLab, on compute-node loopback, with token authentication. It does
not submit a job or start an SSH tunnel. A session-specific connection
script should establish the permitted route to your computer; never reuse
a previous allocation's node, TLS certificate, or token.

The Sphinx site can be built without an allocation:

```bash
source /path/outside/git/mlip-notebook-venv/bin/activate
make html
make livehtml PORT=8766
```

`make livehtml` serves the published pages on the machine's loopback and
rebuilds them when Markdown changes. It is **not** the runnable JupyterLab
server. Forward its port over SSH if viewing it from another computer.

Run the GPU notebooks in order: single MD, batched MD, then the one-rank
LAMMPS example. Execute only one GPU notebook kernel at a time. The reviewed
results page works without a GPU and is safe to inspect before class.
The software-build and multi-GPU episodes are optional; give learners
prepared artifacts when build or queue time would dominate the session.

## Short live demo (webinar)

For a short live demo of Part A, book a one-GPU interactive allocation well
before the session; queue wait is the main risk. Install the pixi
environment and download the checkpoint on a login node beforehand;
compute nodes have no internet access. Set `PS1='$ '` so the prompt shows no user,
host or path. After the run, show `summary.json` rather than scrolling the
log. If the allocation or run fails, show the last saved `summary.json`
from a rehearsal and state that it is a saved result. Serve the lesson pages
locally with `make livehtml`, which binds to `127.0.0.1`, rather than
sharing a remote session.

The webinar used this fallback: Leonardo was in maintenance, so the demo
walked through saved runs.

## Publication

Before publication, perform a privacy review of Markdown, scripts, notebook
outputs, metadata, and rendered HTML. Do not publish usernames, personal
hostnames, project/account numbers, tokens, site-private paths, scheduler
logs, or model weights. Remove notebook outputs generated on a live site;
the source `.md` files, not `.ipynb` exports, are the maintained notebooks.
