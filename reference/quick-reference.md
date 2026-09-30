# Quick reference

| Page | What you do | Runs on | Key command or script |
|---|---|---|---|
| {doc}`../episodes/a1-batched-relaxation` | batched relaxation with TorchSim | NVIDIA GPU (Leonardo); CPU smoke test | `pixi run smoke`, `scripts/test-leonardo-torchsim.sbatch` |
| {doc}`../episodes/a2-orb-models` | Orb-v3, MatGL and D3 on graphite | CPU or NVIDIA GPU | `pixi run graphite-cpu`, `scripts/test-leonardo-orb.sbatch` |
| {doc}`../episodes/a3-matgl-tutorials` | MatGL relaxation, MD, benchmark, training | LUMI, one MI250X GCD | `python examples/matgl/cli.py prefetch`, `scripts/lumi-matgl.sbatch` |
| {doc}`../episodes/a4-training` | fine-tune TensorNet from PBE to r2SCAN | LUMI | `scripts/lumi-training-setup.sh`, `scripts/lumi-training.sbatch` |
| {doc}`../episodes/a5-neb` | CI-NEB of a Li hop in LiFePO4 | LUMI | `python examples/neb/cli.py prefetch`, `scripts/lumi-neb.sbatch` |
| {doc}`../episodes/a6-uma` | Orb-v3, OrbMol and UMA with charge and spin | LUMI | `python examples/uma/cli.py prefetch --model orb` (or `uma`), `scripts/lumi-uma.sbatch` |
| {doc}`../episodes/02-alchemi-image` | build the ALCHEMI image | GH200 site | `scripts/build-alchemi-sif.sh` |
| {doc}`../episodes/03-lammps-mpi` | build LAMMPS with ML-IAP and Kokkos | GH200 site | `scripts/build-lammps-mpi.sbatch` |
| {doc}`../episodes/04-silicon-md` to {doc}`../episodes/06-lammps-replicas` | silicon MD: one trajectory, a batch, LAMMPS replicas | one GH200 | `scripts/run-alchemi.sh`, `scripts/run-lammps.sh` |
| {doc}`../episodes/07-reviewed-results` | read a finished shared-GPU benchmark | no GPU | none |
| {doc}`../episodes/08-scaling` | one simulation on 1, 2 and 4 GPUs | GH200 node | `scripts/lammps-mpi-size-benchmark.sbatch` |

Submit Slurm jobs with `sbatch --account=<PROJECT> <script>`; site set-up is
in {doc}`../setup/index`, variables in {doc}`environment`. UMA weights are
gated: accept the licence on the Hugging Face model page and use a token.
Every page shows saved results, so it can be read without a GPU.
