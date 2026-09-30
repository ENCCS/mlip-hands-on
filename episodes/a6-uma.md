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

# A6 UMA: one model, materials and molecules

:::{objectives}
- Load UMA 1.2 small once and use it for crystals (`omat`) and molecules (`omol`).
- Pass total charge and spin multiplicity to a molecular calculation.
- Compare lattice constants, a spin gap and an ionisation energy with
  their reference data, and measure the cost on one MI250X GCD.
:::

Most universal MLIPs learn one potential energy surface at one level of
theory. UMA [[1](https://arxiv.org/abs/2506.23971)] ([code](https://github.com/facebookresearch/fairchem))
is trained on five datasets at different levels, and a task name tells it
which one to reproduce:

| Task | Dataset | Reference level | Use |
|---|---|---|---|
| `omat` | OMat24 [[2](https://arxiv.org/abs/2410.12771)] | PBE and PBE+U (VASP) | inorganic crystals |
| `omol` | OMol25 [[3](https://arxiv.org/abs/2505.08762)] | ωB97M-V/def2-TZVPD (ORCA) | molecules, with charge and spin |
| `oc20` | OC20 | RPBE | catalysis, adsorbates on surfaces |
| `odac` | ODAC | PBE-D3 | MOFs with guest molecules |
| `omc` | OMC | PBE-D3 | molecular crystals |

The `omol` task is the only one that reads the total charge and spin
multiplicity (`atoms.info["charge"]`, `atoms.info["spin"]`), so the same
geometry can be evaluated as a neutral singlet, a cation or a triplet. The
other tasks ignore them.

## Access

UMA is gated. Accept the licence on the [model page](https://huggingface.co/facebook/UMA),
create a read token in your Hugging Face settings, then log in once on the
login node:

```bash
hf auth login   # stores the token in ~/.cache/huggingface/token
```

## Set-up on LUMI

`fairchem-core` 2.21.0 is the newest release with `uma-s-1p2` that still
pins torch 2.8; later releases pin torch 2.13. The CSC module provides
torch 2.7.1+rocm6.2.4, so fairchem is installed without its dependencies
and the rest from a pinned list that keeps the module's torch. On a login
node, in a copy of this repository at `<SCRATCH>/mlip-uma`:

```bash
module purge && module use /appl/local/csc/modulefiles/ && module load pytorch
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install --no-deps fairchem-core==2.21.0
pip install -r examples/uma/requirements.txt
export HF_HOME=$PWD/hf-home FAIRCHEM_CACHE_DIR=$PWD/fairchem-cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
python examples/uma/cli.py prefetch
sbatch --account=<PROJECT> scripts/lumi-uma.sbatch
```

:::{dropdown} requirements.txt
```{literalinclude} ../examples/uma/requirements.txt
:language: text
```
:::

## Code

One predict unit holds the weights; each task gets a light ASE calculator
on top of it, so switching task does not load the model again.

:::{dropdown} models.py
```{literalinclude} ../examples/uma/models.py
:language: python
:lineno-match:
```
:::

:::{dropdown} structures.py
```{literalinclude} ../examples/uma/structures.py
:language: python
:lineno-match:
```
:::

:::{dropdown} tasks.py
```{literalinclude} ../examples/uma/tasks.py
:language: python
:lineno-match:
```
:::

:::{dropdown} cli.py
```{literalinclude} ../examples/uma/cli.py
:language: python
:start-at: def main
:lineno-match:
```
:::

:::{dropdown} lumi-uma.sbatch
```{literalinclude} ../scripts/lumi-uma.sbatch
:language: bash
:start-at: export PYTHONNOUSERSITE
:lineno-match:
```
:::

## Results

Not yet run: the licence request for the UMA weights is pending.

:::{keypoints}
- One UMA checkpoint covers crystals and molecules; the task name selects
  the level of theory it reproduces.
- The `omol` task takes total charge and spin multiplicity as inputs, so
  ionisation energies and spin gaps come from the same model.
- Compare each task with its own reference level: PBE for `omat`,
  ωB97M-V for `omol`.
:::

## References

1. B. M. Wood et al., UMA: a family of universal models for atoms.
   [arXiv:2506.23971](https://arxiv.org/abs/2506.23971)
2. L. Barroso-Luque et al., OMat24. [arXiv:2410.12771](https://arxiv.org/abs/2410.12771)
3. D. S. Levine et al., OMol25. [arXiv:2505.08762](https://arxiv.org/abs/2505.08762)
