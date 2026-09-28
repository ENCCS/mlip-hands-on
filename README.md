# Molecular dynamics with MACE on GPUs

This repository contains the source of a short, runnable MyST lesson. The
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
pages on loopback. Publishing never runs the GPU cells. The offline results
episode reads only the small checked-in CSV.

Site-specific commands live in `setup/arrhenius.md` and `setup/jupiter.md`.
The JUPITER profile qualifies short native LAMMPS functional checks from one
through eight GPUs, but not ALCHEMI, Jupyter, or scaling performance. See
`reference/instructor.md` before a live session.
