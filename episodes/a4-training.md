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

# Training and fine-tuning with MatGL

:::{objectives}
- Train a small graph network that predicts formation energy.
- Fine-tune a PBE foundation potential on a few hundred r2SCAN structures.
- Compare zero-shot, fine-tuned and from-scratch errors on the same test set.
:::

The previous pages use foundation potentials as they are (zero-shot). The
{doc}`Background <00-background>` section *Pre-train, then fine-tune*
explains when that is not enough: when you need numbers for one system, or
a different level of theory, you fine-tune on a small targeted dataset.
This page does both halves of that recipe with
[MatGL](https://github.com/materialyzeai/matgl) 4.0.3 on one LUMI GPU
(AMD MI250X, one GCD). The code re-implements the MatGL tutorials
*Training a Formation Energy Model* and *Fine-Tuning a M3GNet Potential*.

The example is a small package,
`examples/training`:

| File | Role |
|---|---|
| `prefetch.py` | downloads data and the pretrained model on a login node; writes small subsets |
| `eform.py` | formation-energy model (MEGNet or M3GNet) |
| `finetune.py` | zero-shot, fine-tuned and from-scratch cases |
| `potential.py` | loads, re-references and evaluates TensorNet potentials |
| `fit.py`, `report.py` | Lightning training loop; JSON and figures |

## Formation energy

The data are the Materials Project 2018.6.1 formation energies used for
MEGNet (69 239 crystals). The prefetch step keeps a random subset of 5000
(fixed seed), split 80/10/10 into training, validation and test sets. The
model is the MEGNet of the MatGL tutorial (4 Å graph cutoff, three blocks,
set2set readout); `--model m3gnet` swaps in M3GNet. Targets are
standardised with the training mean and standard deviation, and the
checkpoint with the lowest validation loss is kept.

```bash
python -m training eform --data <SCRATCH>/mlip-training/data \
  --outdir results/eform --epochs 200
```

## Fine-tuning a foundation potential

`TensorNet-PES-MatPES-PBE-2025.2` was trained on PBE energies, forces and
stresses from the MatPES training split. The target here is r2SCAN, a
more accurate meta-GGA functional. The data come from the MatPES r2SCAN
*test* split, which the PBE model never saw: all structures that contain
lithium and at most 64 atoms, up to 1500 of them, split 70/10/20. The test
set is fixed for every case.

| Case | Starting weights | Atom energies | Learning rate |
|---|---|---|---|
| zero-shot (PBE) | pretrained | PBE | none |
| zero-shot, r2SCAN atom energies | pretrained | r2SCAN | none |
| fine-tuned (10, 50, 100%) | pretrained | r2SCAN | 2 × 10⁻⁴ |
| from scratch (10, 50, 100%) | random, same architecture | r2SCAN | 1 × 10⁻³ |

Each trained case runs 150 epochs with a Huber loss on energy per atom,
forces and stress (weight 0.1), a cosine learning-rate decay and seed 42.
The 10% and 50% sets are nested in the full training set.

Two details matter. First, MatGL adds per-element reference energies (the
isolated-atom energies) to the network output. PBE and r2SCAN total
energies differ by several eV/atom, so the second zero-shot row swaps in
the r2SCAN atom energies without changing the network, and both trained
cases use them. Second, the fine-tuned module must keep the pretrained
scaling (`data_std`); the MatGL default of 1.0 silently rescales every
prediction. MatPES stores stress in Voigt order (xx, yy, zz, yz, xz, xy)
and kbar; the prefetch step writes full 3 × 3 tensors, which MatGL 4.0.3
needs for its stress loss.

```{literalinclude} ../examples/training/finetune.py
:language: python
:pyobject: train_case
```

## Run on LUMI

On a login node, build a virtual environment on top of the CSC ROCm
PyTorch module and download everything once (about 1.3 GB):

```bash
export MLIP_LESSON_ROOT=<SCRATCH>/mlip-hands-on
export MLIP_TRAINING_DIR=<SCRATCH>/mlip-training
bash $MLIP_LESSON_ROOT/scripts/lumi-training-setup.sh
```

Then submit one job that runs both examples on one GCD:

```bash
sbatch --account=<PROJECT> --export=ALL \
  $MLIP_LESSON_ROOT/scripts/lumi-training.sbatch
```

The job sets private MIOpen and temporary directories, disables user
site-packages and runs offline. Results go to
`$MLIP_TRAINING_DIR/results-<jobid>/{eform,finetune}` as JSON and PNG.

To check the code on a laptop CPU first, use a few structures and two
epochs:

```bash
cd examples
uv run --with matgl==4.0.3 --with pandas --with matplotlib \
  python -m training prefetch --data training_data --skip-eform --n-pes 60
uv run --with matgl==4.0.3 --with pandas --with matplotlib \
  python -m training finetune --data training_data --accelerator cpu \
  --epochs 2 --fractions 0.5 1.0
```

## Results

:::{note}
The LUMI results will be added here after the run. A two-epoch laptop
check on 60 structures only tests the plumbing: it gave a zero-shot energy
error of about 6000 meV/atom with PBE atom energies and 113 meV/atom with
r2SCAN atom energies.
:::

## Reading the results

- The raw zero-shot energy error measures the difference between the two
  functionals' total energies, not the quality of the model. Compare
  trained cases with the re-referenced zero-shot row.
- Forces do not depend on atom energies, so both zero-shot rows share one
  force error.
- Fine-tuning starts from a good potential energy surface; with few
  structures it should beat training from scratch by a wide margin. The gap
  should shrink as the training set grows.
- A single seed and a single split give one sample. Treat differences of a
  few meV/atom as noise.
- The fine-tuned model is specialised to lithium compounds at r2SCAN level.
  Keep the original model for other chemistry.

:::{keypoints}
- A foundation potential is a strong starting point: fine-tuning on a few
  hundred structures moves it to a new level of theory.
- Change the per-element reference energies when the functional changes,
  and keep the pretrained scaling.
- Evaluate every case on the same held-out test set.
:::

## References

- C. Chen, W. Ye, Y. Zuo, C. Zheng and S. P. Ong, *Graph networks as a
  universal machine learning framework for molecules and crystals*,
  [Chem. Mater. 31, 3564](https://doi.org/10.1021/acs.chemmater.9b01294)
  (2019).
- C. Chen and S. P. Ong, *A universal graph deep learning interatomic
  potential for the periodic table*,
  [Nat. Comput. Sci. 2, 718](https://doi.org/10.1038/s43588-022-00349-3)
  (2022).
- G. Simeon and G. De Fabritiis, *TensorNet: Cartesian tensor
  representations for efficient learning of molecular potentials*,
  [arXiv:2306.06482](https://arxiv.org/abs/2306.06482) (2023).
- A. D. Kaplan et al., *A foundational potential energy surface dataset for
  materials*, [arXiv:2503.04070](https://arxiv.org/abs/2503.04070) (2025).
- T. W. Ko et al., *Materials Graph Library (MatGL), an open-source graph
  deep learning library for materials science and chemistry*,
  [arXiv:2503.03837](https://arxiv.org/abs/2503.03837) (2025).
- J. W. Furness et al., *Accurate and numerically efficient r2SCAN
  meta-generalized gradient approximation*,
  [J. Phys. Chem. Lett. 11, 8208](https://doi.org/10.1021/acs.jpclett.0c02405)
  (2020).
