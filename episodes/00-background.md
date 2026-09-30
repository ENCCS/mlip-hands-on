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

# Background: universal MLIPs

:::{objectives}
- Define a foundation MLIP and its training data.
- Explain why most materials models miss dispersion.
- Choose and check a model and a GPU engine.
:::

Core concepts for the hands-on parts. Numbered references are at the end;
reviews are in {doc}`../reference/reading`.

## Interatomic potentials and MLIPs

![Accuracy versus cost: classical force fields, MLIPs and DFT.](../_static/accuracy-ladder.drawio.png)

- A potential gives energy from atomic positions; its gradient gives the
  forces for molecular dynamics (MD) and relaxation.
- Classical force fields: fixed functional form, fast, limited
  transferability. First-principles (quantum-mechanical) methods such as
  density functional theory (DFT): accurate, cost grows steeply with system
  size.
- An MLIP is trained on first-principles (DFT) energies and forces: near-DFT
  accuracy at near force-field cost. The idea dates from 2007 [[1](https://doi.org/10.1103/PhysRevLett.98.146401)].

## MLIPs among AI methods for materials

![Six families of AI methods for materials, with interatomic potentials highlighted.](../_static/bg-ai-families.drawio.png)

- Six families: electronic structure, interatomic potentials, generative
  models, structure to property, LLM agents, autonomous labs and open data.
- This lesson covers interatomic potentials only.

(background-foundation)=
## From bespoke to foundation models

![Timeline from system-specific MLIPs (2007 to 2022) to foundation MLIPs (2024 to 2026).](../_static/mlip-timeline.drawio.png)

*MLIP milestones, from one model per material to one model reused
everywhere. Own diagram; logos identify the developing organisations.*

- System-specific MLIP: trained for one material, refitted for the next.
- Equivariant graph networks, NequIP [[2](https://doi.org/10.1038/s41467-022-29939-5)] and MACE [[3](https://arxiv.org/abs/2206.07697)], need far less data.
- Foundation MLIP: pre-trained across the periodic table, reused without
  retraining. MACE-MP-0 [[4](https://arxiv.org/abs/2401.00096)] came first; UMA [[7](https://arxiv.org/abs/2506.23971)] and Orb-v3 [[8](https://arxiv.org/abs/2504.06231)] in 2025.
- Coverage follows the data: common elements appear in hundreds of
  thousands of structures, rare ones (noble gases) in a handful (MPtrj
  counts in [[4](https://arxiv.org/abs/2401.00096)]). Check your elements and short-range repulsion before
  screening arbitrary crystals.

Training data grew over a hundredfold in a few years:

| Dataset | Scale | Reference level |
|---|---|---|
| MPtrj [[5](https://doi.org/10.1038/s42256-023-00716-3)] | about 1.58 million configurations, 89 elements | PBE(+U) |
| MatterSim | about 17 million configurations (active learning) | PBE(+U) |
| GNoME | about 89 million structures (not public) | PBE(+U) |
| OMat24 [[6](https://doi.org/10.1038/s43588-026-00996-w)] | about 118 million inorganic structures | PBE+U |
| OMol25 [[9](https://arxiv.org/abs/2505.08762)] | more than 100 million molecular calculations | ωB97M-V/def2-TZVPD |
| UMA training [[7](https://arxiv.org/abs/2506.23971)] | about 500 million structures | mixed |

Materials models learn PBE, which misses dispersion, so {doc}`a2-orb-models`
adds D3. OMol25 models use a different reference level.

## How the models are built

- Message passing: in a graph neural network (GNN), atoms exchange
  information with neighbours over several rounds; features are learnt,
  not hand-designed.
- Equivariant: rotate the structure and the predicted forces rotate with it,
  while the energy is unchanged. NequIP needs up to about 1000 times less
  data [[2](https://doi.org/10.1038/s41467-022-29939-5)].

![Equivariance: rotating the structure rotates the forces; the energy is unchanged.](../_static/equivariance.drawio.png)

- Transformers scale the idea to the largest datasets. By 2026, simpler
  designs compete closely: data and scale matter more than architecture.

## Pre-train, then fine-tune

![Pre-train on a large dataset, use zero-shot, then fine-tune on a small targeted set.](../_static/bg-pretrain-finetune.drawio.png)

- Recipe [[10](https://doi.org/10.1063/5.0299305)]: zero-shot for screening; fine-tune on a small targeted set
  when you need numbers; add data by uncertainty (active learning);
  validate the property, not only the energy.
- Fine-tuning can cause catastrophic forgetting on other systems [16];
  keep the original model for general use.

Fine-tuning is data-efficient [[10](https://doi.org/10.1063/5.0299305)] (errors in meV/atom against DFT; lower
is better):

- High-entropy alloy: fine-tuned 13.8 meV/atom; from scratch 16.4 (MACE)
  and 24.1 (ACE).
- Molybdenum (MACE-MP-0b3): $C_{11}$ error 45.9% zero-shot, 2.6%
  fine-tuned.
- Silicon (MACE-MP-0b): 19-53% zero-shot, 0.6-5.2% fine-tuned.
- Mechanical properties are a known zero-shot weak spot.
- Amorphous materials are another: across 41 universal models, some exceed
  100% relative energy error on amorphous carbon and get ring statistics
  wrong; fine-tuning on only four amorphous SiO2 structures cuts the energy
  error more than 5 times [[17](https://arxiv.org/abs/2607.11384)].
- Our own run, {doc}`a4-training`: TensorNet (MatPES-PBE) fine-tuned to
  r2SCAN on 84 Li structures reaches 60 meV/atom energy and 128 meV/Å force
  error, against 513 and 427 from scratch (one run, one MI250X GCD).

(background-engines)=
## GPU engines

- ASE and LAMMPS run one system at a time; their GPU support targets
  classical force fields.
- TorchSim (PyTorch), kUPS (JAX) and NVIDIA ALCHEMI Toolkit (PyTorch and
  Warp) batch many systems into one GPU call [[11](https://github.com/TorchSim/torch-sim)]. Part A uses TorchSim;
  Part B uses ALCHEMI Toolkit.
- ALCHEMI also supplies CUDA-only kernels (neighbour lists, D3, Ewald) used
  under UMA, Orb, PET and TorchSim, including the D3 on
  {doc}`a2-orb-models`. They run on Leonardo, not on LUMI.

![Engine building blocks: a potential, an integrator and a thermostat combine into different simulation types.](../_static/engine-building-blocks.drawio.png)

*Engines are built from swappable blocks. Own diagram, after the kUPS
design.*

- Leonardo (NVIDIA): CUDA-only kernels such as cuEquivariance and ALCHEMI
  run.
- LUMI (AMD MI250X): ROCm PyTorch runs MACE; CUDA-only kernels do not.
- NequIP and Allegro foundation models run LAMMPS ML-IAP/Kokkos MD on
  both: up to 102.5 million atoms on 256 GPUs, about 44 000 atoms per A100
  and 22 000 per MI250X GCD. NequIP-OAM-XL matches eSEN-30M-OAM on
  Matbench Discovery at about ten times the speed [[12](https://arxiv.org/abs/2607.28461)].
- DPA4C (DeePMD-kit v3.2.0) approaches MACE-OMat accuracy at about 100
  times the throughput; 2.048 billion atoms on 1024 V100 GPUs
  [[18](https://arxiv.org/abs/2608.19041)]. Checkpoints are CC-BY-NC.
- LUMI tip: with the CSC PyTorch module (`torch` 2.7.1+rocm6.2.4),
  `torch.det` and `prod` fail in float32 on the MI250X but work in float64,
  so MatGL runs in float64 there ({doc}`a3-matgl-tutorials`,
  {doc}`a4-training`).
- Pin versions and re-check after upgrades. Bugs that silently gave wrong
  results were fixed between July and September 2026 in cuEquivariance
  v0.12.0 (fused tensor-product reduction), NequIP v0.19.0 (wrong forces
  and stress in TorchSim) and TorchSim v0.6.1 (D3, Ewald, PME and DSF
  stress sign) [[19](https://github.com/TorchSim/torch-sim/releases/tag/v0.6.1)].
- Measured multi-GPU MACE scaling: {doc}`08-scaling`.

(background-choosing)=
## Choosing and trusting a model

![Choosing a model and engine: GPU type, dispersion, fine-tuning, batching.](../_static/bg-choose-model.drawio.png)

- Choose by task and hardware, not by the top leaderboard row.
- Validate the property you study; compare several models.
- Rules of thumb, a check protocol and the evidence behind them:
  {doc}`../reference/choosing-a-model`.

Fast universal models ([Matbench Discovery](https://matbench-discovery.materialsproject.org)
data, accessed 29 September 2026 [[13](https://doi.org/10.1038/s42256-025-01055-1)]):

| Model | Params | F1 ↑ | κSRME ↓ | Licence | Use case |
|---|---:|---:|---:|---|---|
| [Orb-v3](https://github.com/orbital-materials/orb-models) [[8](https://arxiv.org/abs/2504.06231)] | 26M | 0.905 | 0.21 | Apache-2.0 | fast; D3 variant for van der Waals |
| [SevenNet-Omni](https://github.com/MDIL-SNU/SevenNet) | 55M | 0.906 | 0.19 | MIT | D3 built in; LAMMPS and TorchSim |
| [NequIP-OAM-XL](https://github.com/mir-group/nequip) [[12](https://arxiv.org/abs/2607.28461)] | 32M | 0.906 | 0.13 | MIT / CC-BY | also runs on AMD GPUs (LUMI) |
| [MatRIS-10M-OAM](https://github.com/HPC-AI-Team/MatRIS) | 10M | 0.921 | 0.22 | BSD-3 | best accuracy for its size |
| [MatterSim v1 5M](https://github.com/microsoft/mattersim) | 4.5M | 0.862 | 0.57 | MIT | small and fast |
| [EquiformerV3-OAM](https://github.com/atomicarchitects/equiformer_v3) | 30M | 0.931 | 0.12 | MIT | accuracy leader, slower |

- F1 (0 to 1, higher is better): stable-crystal classification on
  [Matbench Discovery](https://matbench-discovery.materialsproject.org) [[13](https://doi.org/10.1038/s42256-025-01055-1)].
  κSRME (lower is better): thermal-conductivity error.
- The leaderboard moves within months; OMat24-trained models reach F1 of
  about 0.92 to 0.93.
- Most GPU speed-ups are NVIDIA-only (Leonardo). On AMD (LUMI), choose a
  pure-PyTorch model such as [NequIP](https://github.com/mir-group/nequip)
  or [MACE](https://github.com/ACEsuit/mace).
- A low energy error does not guarantee stable MD. Benchmark your property
  class and check stability.
- One error number is not enough: in finite-temperature MD of 15 foundation
  MLIPs, the tier with the lowest force error had the worst median pressure
  error (3.40 GPa) [[20](https://arxiv.org/abs/2607.03433)]. It is now the
  Matbench Discovery MD task.
- Beyond one score: [MLIP Arena](https://github.com/atomind-ai/mlip-arena)
  [[14](https://arxiv.org/abs/2509.20630)] tests equations of state, phonons, diffusion barriers and diatomic
  curves; [mlipbenchmarks](https://github.com/peastman/mlipbenchmarks) [[15](https://doi.org/10.1021/acs.jctc.6c00130)]
  measures accuracy, MD speed, GPU memory and stability.
- Ensemble: run several models; disagreement flags low confidence.
- Check speed and GPU memory at your system size.
- PBE-trained models miss dispersion. Grimme's
  [D3 correction](https://github.com/dftd3/simple-dftd3) adds it on the GPU
  in [TorchSim](https://github.com/TorchSim/torch-sim) and `orb-models`
  (used on {doc}`a2-orb-models`).

## Outlook

- Directions: long-range electrostatics without charge labels, learned DFT
  functionals for better reference data, generative models screened with
  MLIPs, language-model agents driving simulation codes.
- Early agentic workflows: in an NVIDIA test, coding agents wrote 45
  batched GPU MLIP pipelines, but none pushed back on an ill-posed task, and
  an unspecified thermostat changed Li diffusion by 3 to 5 times
  [[21](https://developer.nvidia.com/blog/how-ai-coding-agents-can-unlock-materials-simulation-with-nvidia-alchemi-toolkit/)].
  Physics choices still need an expert.
- In every case, validate the property you care about.

> "A poorly posed initial question results in AI scientific slop, an
> unfortunate side effect that is now becoming far too common."
> Shyue Ping Ong, 2026 [[22](https://www.materialyze.ai/post/the-non-ai-pocalypse-in-materials-science)]

:::{keypoints}
- MLIPs learn DFT energies and forces at near force-field cost.
- Use foundation models zero-shot; fine-tune for quantitative accuracy.
- PBE-trained models miss dispersion without D3.
- Batched GPU engines run many systems in one call; choose a model by task
  and validate the property you study.
- One error number is not enough; pin package versions.
:::

## References

1. J. Behler, M. Parrinello, Phys. Rev. Lett. 98, 146401 (2007).
   [doi:10.1103/PhysRevLett.98.146401](https://doi.org/10.1103/PhysRevLett.98.146401)
2. S. Batzner et al., Nat. Commun. 13, 2453 (2022).
   [doi:10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5)
3. I. Batatia et al., MACE. [arXiv:2206.07697](https://arxiv.org/abs/2206.07697)
4. I. Batatia et al., A foundation model for atomistic materials chemistry.
   [arXiv:2401.00096](https://arxiv.org/abs/2401.00096)
5. B. Deng et al., CHGNet and MPtrj, Nat. Mach. Intell. 5, 1031 (2023).
   [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3)
6. L. Barroso-Luque et al., OMat24, Nat. Comput. Sci. 6, 642 (2026). [doi:10.1038/s43588-026-00996-w](https://doi.org/10.1038/s43588-026-00996-w); [arXiv:2410.12771](https://arxiv.org/abs/2410.12771)
7. B. M. Wood et al., UMA. [arXiv:2506.23971](https://arxiv.org/abs/2506.23971)
8. B. Rhodes et al., Orb-v3. [arXiv:2504.06231](https://arxiv.org/abs/2504.06231)
9. D. S. Levine et al., OMol25. [arXiv:2505.08762](https://arxiv.org/abs/2505.08762)
10. X. Liu et al., J. Appl. Phys. 139, 041101 (2026).
    [doi:10.1063/5.0299305](https://doi.org/10.1063/5.0299305); companion
    study [arXiv:2506.07401](https://arxiv.org/abs/2506.07401)
11. [TorchSim](https://github.com/TorchSim/torch-sim),
    [kUPS](https://github.com/cusp-ai-oss/kups),
    [ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit)
12. S. R. Kavanagh et al., NequIP and Allegro foundation models.
    [arXiv:2607.28461](https://arxiv.org/abs/2607.28461)
13. J. Riebesell et al., Matbench Discovery, Nat. Mach. Intell. 7, 836 (2025).
    [doi:10.1038/s42256-025-01055-1](https://doi.org/10.1038/s42256-025-01055-1);
    [leaderboard](https://matbench-discovery.materialsproject.org)
14. Chiang et al., MLIP Arena, NeurIPS 2025 Datasets and Benchmarks.
    [arXiv:2509.20630](https://arxiv.org/abs/2509.20630)
15. Eastman, Pretti and Markland, mlipbenchmarks, J. Chem. Theory Comput. 22, 6108 (2026).
    [doi:10.1021/acs.jctc.6c00130](https://doi.org/10.1021/acs.jctc.6c00130)
16. Kim et al., catastrophic forgetting,
    npj Comput. Mater. 12, 26 (2026).
17. Fragapane and Deringer, AM26 amorphous-materials benchmark.
    [arXiv:2607.11384](https://arxiv.org/abs/2607.11384)
18. DPA4 and DPA4C, [DeePMD-kit v3.2.0](https://github.com/deepmodeling/deepmd-kit/releases/tag/v3.2.0).
    [arXiv:2608.19041](https://arxiv.org/abs/2608.19041)
19. Release notes:
    [cuEquivariance v0.12.0](https://github.com/NVIDIA/cuEquivariance/releases/tag/v0.12.0),
    [NequIP v0.19.0](https://github.com/mir-group/nequip/releases/tag/v0.19.0),
    [TorchSim v0.6.1](https://github.com/TorchSim/torch-sim/releases/tag/v0.6.1)
20. Dyna-Mat, foundation MLIPs in finite-temperature MD.
    [arXiv:2607.03433](https://arxiv.org/abs/2607.03433)
21. NVIDIA, How AI coding agents can unlock materials simulation with NVIDIA
    ALCHEMI Toolkit (2026).
    [blog](https://developer.nvidia.com/blog/how-ai-coding-agents-can-unlock-materials-simulation-with-nvidia-alchemi-toolkit/)
22. S. P. Ong, The Non-AI-pocalypse in Materials Science (2026).
    [post](https://www.materialyze.ai/post/the-non-ai-pocalypse-in-materials-science)
