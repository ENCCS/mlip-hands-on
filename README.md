# Molecular dynamics with MACE on GPUs

This standalone ENCCS lesson uses MyST Markdown for both its web pages and
JupyterLab notebooks. `content/` is the only lesson source. `examples/` has
small Python programs, LAMMPS inputs, and starting structures; `scripts/`
has the commands participants run. Optional build and validation helpers
belong in `maintainer/` and are not needed for the examples.

The branch `lesson/minimal-md-cli` develops a CLI-based LAMMPS route and a
short ALCHEMI route on Arrhenius and JUPITER GH200 nodes. It does not change
the published `main` lesson until separately reviewed and merged. Model
weights, SIFs, native binaries, site credentials, generated notebooks, and
results are deliberately absent from Git.

Install `requirements.txt` into a private Python environment, then run
`make html`. The strict Sphinx build leaves GPU cells unexecuted. To view a
local live build, run `make livehtml PORT=8766` and forward that loopback
port over SSH. Start at [`content/index.md`](content/index.md).

Historical Python-LAMMPS timing results are not measurements of this
branch's LAMMPS CLI workflow. Requalify each site before adding performance
claims.
