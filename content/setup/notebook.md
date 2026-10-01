# Open the MyST notebook

The `.md` pages are the notebooks. Use a private JupyterLab environment
with Jupytext and the lesson's MyST renderer. Start the server inside
an allocated GPU job and open its authenticated URL through your site's
approved SSH forwarding route. Do not share the token or expose a public
listener.

Keep model weights, SIFs, the notebook token, and Slurm logs outside Git.
In a GPU-capable site shell, prepare a private environment file from
[`.env.example`](../../.env.example). Give it owner-only permissions, then
set its path and the lesson checkout before submitting one bounded notebook
job:

```bash
export MLIP_LESSON_ROOT="$PWD"
export MLIP_ENV_FILE=/path/to/private/mlip-lesson.env
export MLIP_NOTEBOOK_VENV=/path/to/private/jupyter-venv
export MLIP_JOB_LOG_DIR=/path/to/private/notebook-logs
export MLIP_ACCOUNT=YOUR_CURRENT_ACCOUNT
bash scripts/submit-arrhenius-jupyter.sh  # or submit-jupiter-jupyter.sh
```

The submit helper reports one job ID and a private token-log path. Do not
resubmit after an uncertain response; inspect the scheduler first. Arrhenius
may use a currently authorized `MLIP_RESERVATION`; JUPITER has a separate
Booster job file. Both start JupyterLab inside one GPU allocation with TLS.

On your laptop, from its own checkout of this lesson, connect through the
site's configured SSH login alias and the reported job ID. For example,
replace `SITE_LOGIN_ALIAS` with your alias and `JOB_ID` with the submitted job:

```bash
bash scripts/connect-from-laptop.sh notebook SITE_LOGIN_ALIAS JOB_ID
```

The connector checks the running allocation and the server certificate
fingerprint, then forwards to laptop loopback. Read the token URL from the
private job log using your authenticated site login, and use the connector's
local port in that URL. Keep the forwarding terminal open while using the
notebook. When finished, close the tunnel and cancel the notebook job:

```bash
scancel JOB_ID
```

Open a lesson `.md` as a Jupytext notebook. The single-trajectory ALCHEMI
page loads `scripts/notebook_alchemi.py` once, then runs its visible Python
cell inside the selected SIF. Each LAMMPS cell sources
`scripts/${MLIP_SITE}-lammps-env.sh` in its own Bash process before the
visible `lmp` command. On Arrhenius, the notebook job extracts the reviewed
native Python archive into job-local scratch and passes its path to those
cells; on JUPITER, use the selected native Python environment in the
private `.env`. Published HTML never executes GPU cells.
