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

## What the job computes

- `omat`: relax the conventional cells of Si (diamond) and Cu (fcc),
  positions and cell together, with BFGS to fmax 0.005 eV/Å.
- `omol`, spin: relax O<sub>2</sub> as a triplet (`spin=3`, the ground
  state) and as a singlet (`spin=1`); the gap is the difference of the two
  minima.
- `omol`, charge: relax neutral water (`charge=0, spin=1`), then evaluate
  the cation (`charge=1, spin=2`) at that geometry for the vertical
  ionisation energy, and relax it for the adiabatic one.
- Cost: model load time, the first force call and the median of 20 later
  calls, in float32 and float64, one process per precision.

UMA computes stress from the cell determinant (`torch.det`), which fails in
float32 on LUMI's torch 2.7.1+rocm6.2.4 (see {doc}`a3-matgl-tutorials`), so
the science runs in float64 through `base_precision_dtype`. The float32
timing run records which calls fail rather than stopping.

### Reference values

| Quantity | Reference | Value |
|---|---|---|
| Si lattice constant | experiment, 22.5 °C (CODATA 2022) [[4](https://physics.nist.gov/cgi-bin/cuu/Value?asil)] | 5.431 Å |
| Si lattice constant | PBE, all-electron, static lattice [[5](https://doi.org/10.1126/science.aad3000)] | 5.470 Å |
| Cu lattice constant | experiment, 25 °C [[6](https://doi.org/10.1107/S0567739469001549)] | 3.615 Å |
| Cu lattice constant | PBE, all-electron, static lattice [[5](https://doi.org/10.1126/science.aad3000)] | 3.629 Å |
| O<sub>2</sub> a¹Δ<sub>g</sub> minus X³Σ<sub>g</sub><sup>−</sup>, T<sub>e</sub> | experiment [[7](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1000)] | 7918.1 cm<sup>−1</sup> = 0.982 eV |
| H<sub>2</sub>O first ionisation energy | experiment, evaluated [[8](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=20)] | 12.621 eV (photoelectron vertical value 12.62 eV) |

The PBE lattice constants come from the equilibrium volumes of the
WIEN2k reference set ([data](https://github.com/molmod/DeltaCodesDFT/blob/master/WIEN2k.txt)):
a = (8V<sub>0</sub>)<sup>1/3</sup> for Si and (4V<sub>0</sub>)<sup>1/3</sup>
for Cu. The `omat` task reproduces PBE, so compare it first with PBE; the
difference from experiment is mostly the functional, plus thermal
expansion and zero-point motion, which a static relaxation leaves out.

The singlet needs care. The a¹Δ<sub>g</sub> state of O<sub>2</sub> is not
a single closed-shell determinant, and OMol25 ran main-group singlets in
restricted Kohn-Sham unless bonds were expected to break
[[3](https://arxiv.org/abs/2505.08762)]. The `omol` singlet is therefore
a closed-shell ωB97M-V state, not the experimental a¹Δ<sub>g</sub>, and a
gap larger than 0.98 eV is expected from the reference method itself.

## Results

Not yet run: access to the gated UMA weights is still pending. The job
writes `examples/uma/results/lattice.csv`, `molecules.json`,
`timing_float32.csv`, `timing_float64.csv` and `versions.json`, and the
tables here will be filled from those files.

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
4. CODATA 2022, lattice parameter of silicon, NIST.
   [physics.nist.gov](https://physics.nist.gov/cgi-bin/cuu/Value?asil)
5. K. Lejaeghere et al., Reproducibility in density functional theory
   calculations of solids, Science 351, aad3000 (2016).
   [doi:10.1126/science.aad3000](https://doi.org/10.1126/science.aad3000)
6. M. E. Straumanis and L. S. Yu, Acta Cryst. A 25, 676 (1969).
   [doi:10.1107/S0567739469001549](https://doi.org/10.1107/S0567739469001549)
7. NIST Chemistry WebBook, O<sub>2</sub>, constants of diatomic molecules.
   [webbook.nist.gov](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1000)
8. NIST Chemistry WebBook, H<sub>2</sub>O, gas-phase ion energetics.
   [webbook.nist.gov](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=20)
