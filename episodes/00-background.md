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

This page summarises the concepts behind the hands-on parts, condensed from
the ENCCS and Sweden AI Factory webinar on universal MLIPs on HPC. Numbered references are listed at
the end; review articles are collected in {doc}`../reference/reading`.

## Interatomic potentials and MLIPs

An interatomic potential gives the energy of a system as a function of the
atomic positions, the potential energy surface. Its gradient gives the
forces that drive molecular dynamics (MD) and structure relaxation.

- Classical force fields use a fixed, hand-crafted functional form. They are
  fast, but accuracy and transferability are limited.
- Quantum methods such as density functional theory (DFT) compute the energy
  from the electronic structure. They are accurate, but the cost grows
  steeply with system size.

A machine-learned interatomic potential (MLIP) replaces the hand-crafted form
with a flexible model trained on quantum reference data (structures with
their energies and forces). The aim is DFT-quality energies and forces at a
cost close to that of a force field. The idea dates from the
high-dimensional neural-network potentials of 2007 [1].

## MLIPs among AI methods for materials

MLIPs are one of six families of methods, which differ in what the model
learns or does:

- **Electronic structure**: neural wavefunctions and learned functionals.
- **Interatomic potentials**: energies and forces learned from DFT data; the
  topic of this lesson.
- **Generative models**: propose new structures, for example diffusion models
  for crystals.
- **Structure to property**: map a structure directly to one property.
- **LLMs and agents**: read, plan and run simulation tools.
- **Autonomous labs and open data**: predict, make and measure.

## From bespoke to foundation models

![Timeline from system-specific MLIPs (2007 to 2022) to foundation MLIPs (2024 to 2026).](../_static/mlip-timeline.drawio.png)

*MLIP milestones, from one model per material to one model reused
everywhere. Own diagram; logos identify the developing organisations.*

A system-specific MLIP is trained for one material and refitted for the next.
Equivariant graph neural networks such as NequIP [2] and MACE [3] learn from
far less data. A universal, or foundation, MLIP is pre-trained across most of
the periodic table and reused without retraining. MACE-MP-0 [4] was among the
first; UMA [7] and Orb-v3 [8] followed in 2025.

Training data grew by more than a hundredfold in a few years: MPtrj has
about 1.58 million configurations [5], OMat24 about 118 million inorganic
structures [6], OMol25 more than 100 million molecular calculations [9], and
UMA was trained on about 500 million structures [7].

## Pre-train, then fine-tune

A foundation model can be used as it is (zero-shot) for screening and
exploration. For quantitative accuracy on one system, it is fine-tuned on a
small, targeted dataset [10]:

- For a high-entropy alloy, a fine-tuned model reached 13.8 meV/atom,
  compared with 16.4 meV/atom (MACE) and 24.1 meV/atom (ACE) trained from
  scratch.
- For molybdenum (MACE-MP-0b3), the error in the elastic constant $C_{11}$
  fell from 45.9% zero-shot to 2.6% after fine-tuning.
- For silicon (MACE-MP-0b), errors fell from 19-53% zero-shot to 0.6-5.2%
  after fine-tuning.

Mechanical properties are a known weak spot of zero-shot models. Fine-tuning
can also cause catastrophic forgetting, so keep the original model for
general use.

## GPU engines

ASE and LAMMPS were designed to run one system at a time, and their GPU
acceleration targets classical force fields. Neural-network potentials
benefit from batched inference on a GPU. TorchSim (PyTorch), kUPS (JAX) and
NVIDIA ALCHEMI Toolkit (PyTorch and Warp) batch many systems into one GPU
call [11]. Part A uses TorchSim for this; Part B uses ALCHEMI Toolkit.

![Engine building blocks: a potential, an integrator and a thermostat combine into different simulation types.](../_static/engine-building-blocks.drawio.png)

*Engines are built from swappable blocks. Own diagram, after the kUPS
design.*

Leonardo has NVIDIA GPUs, where CUDA-only kernels such as cuEquivariance
and ALCHEMI run. On AMD GPUs such as LUMI's
MI250X, ROCm PyTorch runs MACE, but the CUDA-only kernels do not. NequIP and
Allegro foundation models run LAMMPS MD on both GPU types [12].

## Choosing and trusting a model

- Matbench Discovery ranks models on crystal stability; its headline score,
  F1, runs from 0 to 1 [13].
- The leaderboard moves within months: models trained on OMat24 and other
  data reach F1 of about 0.92 to 0.93. Choose by your task, not by the top
  row of the table.
- A low energy error does not guarantee a stable MD trajectory. Benchmark the
  property class you study, compare several models, and check stability.
- Most universal MLIPs are trained on PBE data, which misses dispersion.
  Grimme's D3 correction adds it, and runs on the GPU in TorchSim.

## Outlook

Current directions include long-range electrostatics learned without charge
labels, learned DFT functionals that provide better reference data,
generative models whose candidates are screened with MLIPs, and early
language-model agents that drive simulation codes. In every case,
validate the property you care about.

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
6. L. Barroso-Luque et al., OMat24. [arXiv:2410.12771](https://arxiv.org/abs/2410.12771)
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
    [doi:10.1038/s42256-025-01055-1](https://doi.org/10.1038/s42256-025-01055-1)
