# Molecular dynamics with MACE on GPUs

This independent Git repository is the maintained source of the
MLIP MyST/Jupyter lesson. The older in-tree lesson in `mlip-hands-on` is
historical and should not receive new edits. This repository does not yet
have a published lesson site. Its configured `origin` is
`git@github.com:ENCCS/mlip-hands-on.git`; pushing and enabling GitHub Pages
require separate review.

It contains the source of a short, runnable MyST lesson. The
same Markdown files build a web handout and open as notebooks in JupyterLab.
The lesson uses one pinned MACE checkpoint to run silicon molecular dynamics
with NVIDIA ALCHEMI Toolkit and LAMMPS ML-IAP/Kokkos.

Start at `index.md`. The core exercises use one GPU; the later scaling
episode is optional. Software builds are included as episodes, but a class
can use artifacts prepared beforehand. No model weights, container images,
site credentials, results directory, or personal environment file are stored
in Git.

To build the pages in a Python environment with `requirements.txt` installed,
run `make html`. `make livehtml PORT=8766` watches the Markdown and serves the
pages on loopback. Published pages never execute GPU MD cells. During the build,
the offline results episode reads the small checked-in CSV to render its table
and figure.

GitHub Actions checks shell and Python examples, scans tracked lesson text for
known private site values, and runs the strict Sphinx HTML build on pull
requests and pushes to `main`. The HTML is retained as a workflow artifact.
There is no Pages deployment job yet: confirm the ENCCS organization plan,
Pages policy, and intended public/private visibility before enabling it.
The source scan is a guardrail; publication still needs human review of the
rendered pages and files.

Site-specific commands live in `setup/arrhenius.md`, `setup/jupiter.md`, and
`setup/leonardo.md`.
The JUPITER profile qualifies short native LAMMPS ML-IAP and Metatomic
functional checks from one through eight GPUs, native ALCHEMI one-GPU
single/batched smokes, and a bounded MyST notebook/server check. These do
not establish scientific agreement or scaling performance. See
`reference/instructor.md` before a live session.
