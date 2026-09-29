# Instructor guide

Prepare and test artifacts before class. Keep the private model, SIF,
LAMMPS build, Jupyter token, and Slurm logs outside the lesson repository.
Start one bounded GPU notebook allocation, verify authenticated access,
and stop it after the session.

The pages and `scripts/` are the participant route. Optional hash,
runtime, allocation, and output checks are in `maintainer/`. Qualify a
release by running participant commands from a clean checkout without
that directory.

Use [model.toml](model.toml) for the pinned model identity. Before a site
run, check the account, partition, reservation, GPU allocation, artifact
identities, wall time, and output target. Never repeat an uncertain Slurm
submission merely because its response was lost.

The private JupyterLab renderer in `jupyterlab-enccs/` supports this
lesson's callouts, tabs and relative excerpts, not every Sphinx directive.
Inspect published HTML and notebook views after a directive change.
