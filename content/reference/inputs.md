# Inputs and paths

Copy `.env.example` to a private file outside the checkout as shown in
[Before you start](../setup/index.md), then edit these paths for your site.
No account, model weight, SIF, executable or token belongs in Git.

| Variable | What it names |
| --- | --- |
| `MLIP_MODEL` | Original trusted MACE checkpoint for ALCHEMI |
| `MLIP_MLIAP_MODEL` | ML-IAP export of that checkpoint for LAMMPS |
| `MLIP_ALCHEMI_SIF` | ARM/GH200 ALCHEMI image |
| `MLIP_NATIVE_PREFIX` | Site-native LAMMPS install prefix containing `bin/lmp` |
| `MLIP_NATIVE_PYTHON` | Matching MACE Python environment on either site |
| `MLIP_LMP` | LAMMPS executable, set by the site environment script |
| `MLIP_ARTIFACT_ROOT` | Parent of your private model and runtime files; also used by the replica example for its fresh output directory |
| `MLIP_NATIVE_RUNTIME_ARCHIVE` | Optional Arrhenius archive unpacked into private compute-node scratch by the Jupyter job |

The scripts take the lesson root as their first argument. From that root,
`"$PWD"` supplies it. Relaxation starts are tracked under
`examples/starts/`.

Load the private file with `set -a; source "$MLIP_ENV_FILE"; set +a`. Then
source the site-specific LAMMPS environment script in the same shell that
runs LAMMPS. Start the Jupyter server with the prepared environment so its
notebook shell cells inherit these variables.
