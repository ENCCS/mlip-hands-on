# Inputs and paths

Copy `.env.example` to a private `.env`, then edit these paths for your
allocation. No account, model weight, SIF, executable or token belongs in Git.

| Variable | What it names |
| --- | --- |
| `MLIP_MODEL` | Original trusted MACE checkpoint for ALCHEMI |
| `MLIP_MLIAP_MODEL` | ML-IAP export of that checkpoint for LAMMPS |
| `MLIP_ALCHEMI_SIF` | ARM/GH200 ALCHEMI image |
| `MLIP_NATIVE_PREFIX` | Site-native LAMMPS install prefix containing `bin/lmp` |
| `MLIP_NATIVE_PYTHON` | Matching MACE Python environment on either site |
| `MLIP_LMP` | LAMMPS executable, set by the site environment script |

The scripts take the lesson root as their first argument. From that root,
`"$PWD"` supplies it. Relaxation starts are tracked under
`examples/starts/`.

Load `.env` with `set -a; . ./.env; set +a`. Then source the site-specific
LAMMPS environment script in the same shell that runs LAMMPS. A Jupyter
server should be started with the prepared environment, so its notebook
shell cells inherit these variables.
