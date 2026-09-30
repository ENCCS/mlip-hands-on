# Universal MLIPs on HPC: hands-on

This independent Git repository is the maintained source of the
MLIP MyST/Jupyter lesson. The older in-tree lesson in `mlip-hands-on` is
historical and should not receive new edits. The public source is
[ENCCS/mlip-hands-on](https://github.com/ENCCS/mlip-hands-on), and the
[lesson pages](https://enccs.github.io/mlip-hands-on/) are built from
the `main` branch by GitHub Actions.

It contains the source of a short, runnable MyST lesson. The
same Markdown files build a web handout and open as notebooks in JupyterLab.
Part A covers screening and training with universal models: batched
relaxation with TorchSim (A1, `examples/torchsim/`), Orb-v3 and D3 on graphite
(A2), MatGL on LUMI (A3, `examples/matgl/`), fine-tuning TensorNet (A4,
`examples/training/`), CI-NEB of a Li hop in LiFePO4 (A5, `examples/neb/`) and
Orb-v3, OrbMol and UMA on crystals and on molecules with charge and spin (A6,
`examples/uma/`). Part B uses one pinned MACE checkpoint to run silicon
molecular dynamics with NVIDIA ALCHEMI Toolkit and LAMMPS ML-IAP/Kokkos.
The webinar slides are on the `slides` page (PDF in `_static/slides/`).

Start at `index.md`. The core exercises use one GPU; the later scaling
episode is optional. Software builds are included as episodes, but a class
can use artifacts prepared beforehand. Small saved results (CSV and JSON) are
committed so that the pages read without a GPU. No model weights, container
images, site credentials or personal environment files are stored in Git.
UMA weights are gated: accept the licence on the Hugging Face model page and
log in with a token before running A6 with `--model uma`.

To build the pages in a Python environment with `requirements.txt` installed,
run `make html`. `make livehtml PORT=8766` watches the Markdown and serves the
pages on loopback. Published pages never execute GPU MD cells. During the build,
the offline results episode reads the small checked-in CSV to render its table
and figure.

GitHub Actions checks shell and Python examples, scans tracked lesson text for
known private site values, and runs the strict Sphinx HTML build on pull
requests and pushes to `main`. The HTML is retained as a workflow artifact.
A successful `main` push also publishes that exact build to GitHub Pages;
pull requests never deploy.
The source scan is a guardrail; publication still needs human review of the
rendered pages and files.

Site-specific commands live in `setup/arrhenius.md`, `setup/jupiter.md`, and
`setup/leonardo.md`.
The JUPITER profile qualifies short native LAMMPS ML-IAP and Metatomic
functional checks from one through eight GPUs, native ALCHEMI one-GPU
single/batched smokes, and a bounded MyST notebook/server check. These do
not establish scientific agreement or scaling performance. See
`reference/instructor.md` before a live session.
