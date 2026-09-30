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

# A6 One model for materials and molecules: Orb and UMA

:::{objectives}
- Run the same checks on crystals and on molecules with Orb-v3 and
  OrbMol, and with UMA 1.2 small.
- Pass total charge and spin multiplicity to a molecular calculation.
- Compare lattice constants, a spin gap and ionisation energies with
  their reference data, and measure the cost on one MI250X GCD.
:::

Most universal MLIPs learn one potential energy surface at one level of
theory. Two open routes cover both crystals and molecules:

- Orb [[9](https://arxiv.org/abs/2504.06231)] ([code](https://github.com/orbital-materials/orb-models)):
  one architecture, separate checkpoints. Orb-v3 trained on OMat24 for
  crystals; OrbMol [[10](https://www.orbitalindustries.com/posts/orbmol-extending-orb-to-molecular-systems)]
  trained on OMol25 for molecules, with total charge and spin as inputs.
  Weights are public.
- UMA [[1](https://arxiv.org/abs/2506.23971)] ([code](https://github.com/facebookresearch/fairchem)):
  one checkpoint trained on five datasets; a task name selects the level
  of theory it reproduces. Weights are gated (licence approval on Hugging Face).

| Task | Dataset | Reference level | Orb checkpoint | Use |
|---|---|---|---|---|
| `omat` | OMat24 [[2](https://arxiv.org/abs/2410.12771)] | PBE and PBE+U (VASP) | `orb-v3-conservative-inf-omat` | inorganic crystals |
| `omol` | OMol25 [[3](https://arxiv.org/abs/2505.08762)] | ωB97M-V/def2-TZVPD (ORCA) | `orb-v3-conservative-omol` (OrbMol) | molecules, with charge and spin |
| `oc20` | OC20 | RPBE | none | catalysis, adsorbates on surfaces |
| `odac` | ODAC | PBE-D3 | none | MOFs with guest molecules |
| `omc` | OMC | PBE-D3 | none | molecular crystals |

OrbMol and the UMA `omol` task read the total charge and spin
multiplicity from `atoms.info["charge"]` and `atoms.info["spin"]`, so the
same geometry can be evaluated as a neutral singlet, a cation or a
triplet. Both use the multiplicity 2S+1 (the orb-models 0.5.5 README
example still shows 0; the conditioner treats 0 as unset). The crystal
models ignore both keys.

## Set-up on LUMI

The CSC module provides Python 3.11 and torch 2.7.1+rocm6.2.4.

- orb-models 0.5.5 is the last release for Python 3.11 and torch 2.7; it
  has Orb-v3 and the first OrbMol. Releases from 0.6 need Python 3.12 and
  torch 2.8 or later. OrbMol-v2 (`pretrained.orbmol_v2`, orb-models 0.7,
  long-range electrostatics through `nvalchemiops`) is therefore not used
  here.
- `fairchem-core` 2.21.0 is the newest release with `uma-s-1p2` that still
  pins torch 2.8; later releases pin torch 2.13. It is installed without
  its dependencies, and the rest from a pinned list that keeps the
  module's torch.

On a login node, in a copy of this repository at `<SCRATCH>/mlip-uma`:

```bash
module purge && module use /appl/local/csc/modulefiles/ && module load pytorch
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install --no-deps fairchem-core==2.21.0
pip install -r examples/uma/requirements.txt
export HF_HOME=$PWD/hf-home FAIRCHEM_CACHE_DIR=$PWD/fairchem-cache CACHED_PATH_CACHE_ROOT=$PWD/cached-path
python examples/uma/cli.py prefetch --model orb
sbatch --account=<PROJECT> scripts/lumi-uma.sbatch orb
```

:::{dropdown} requirements.txt
```{literalinclude} ../examples/uma/requirements.txt
:language: text
```
:::

### UMA access

UMA is gated. Accept the licence on the [model page](https://huggingface.co/facebook/UMA),
create a read token in your Hugging Face settings, log in once on the
login node, then prefetch and submit with `uma`:

```bash
hf auth login   # stores the token in ~/.cache/huggingface/token
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
python examples/uma/cli.py prefetch --model uma
sbatch --account=<PROJECT> scripts/lumi-uma.sbatch uma
```

## Code

Each backend exposes `calculator(task)` for `omat` or `omol`. Orb loads
one checkpoint per task; UMA loads one predict unit and puts a light ASE
calculator on top of it for each task. The tasks do not know which
backend they run.

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
  minima. As a control for the charge input, evaluate the O<sub>2</sub><sup>+</sup>
  doublet (`charge=1, spin=2`) at the triplet geometry.
- `omol`, charge: relax neutral water (`charge=0, spin=1`), then evaluate
  the cation (`charge=1, spin=2`) at that geometry for the vertical
  ionisation energy, and relax it for the adiabatic one.
- Cost: model load time, the first force call and the median of 20 later
  calls, in float32 and float64, one process per precision.

ASE decides whether to recompute from positions, cell and atomic numbers,
not from `atoms.info`. A new charge or spin on the same geometry would
return the cached energy, so `tasks.attach` resets the calculator first.

Stress on a periodic cell uses the cell determinant (`torch.det`), which
fails in float32 on LUMI's torch 2.7.1+rocm6.2.4 (see
{doc}`a3-matgl-tutorials`), so the science runs in float64. The float32
timing run records which calls fail rather than stopping.

### Reference values

| Quantity | Reference | Value |
|---|---|---|
| Si lattice constant | experiment, 22.5 °C (CODATA 2022) [[4](https://physics.nist.gov/cgi-bin/cuu/Value?asil)] | 5.431 Å |
| Si lattice constant | PBE, all-electron, static lattice [[5](https://doi.org/10.1126/science.aad3000)] | 5.470 Å |
| Cu lattice constant | experiment, 25 °C [[6](https://doi.org/10.1107/S0567739469001549)] | 3.615 Å |
| Cu lattice constant | PBE, all-electron, static lattice [[5](https://doi.org/10.1126/science.aad3000)] | 3.629 Å |
| O<sub>2</sub> a¹Δ<sub>g</sub> minus X³Σ<sub>g</sub><sup>−</sup>, T<sub>e</sub> | experiment [[7](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1000)] | 7918.1 cm<sup>−1</sup> = 0.982 eV |
| O<sub>2</sub> first ionisation energy | experiment, evaluated, adiabatic [[11](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=20)] | 12.07 eV |
| H<sub>2</sub>O first ionisation energy | experiment, evaluated [[8](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=20)] | 12.621 eV (photoelectron vertical value 12.62 eV) |

The PBE lattice constants come from the equilibrium volumes of the
WIEN2k reference set ([data](https://github.com/molmod/DeltaCodesDFT/blob/master/WIEN2k.txt)):
a = (8V<sub>0</sub>)<sup>1/3</sup> for Si and (4V<sub>0</sub>)<sup>1/3</sup>
for Cu. The crystal models reproduce PBE, so compare them first with PBE;
the difference from experiment is mostly the functional, plus thermal
expansion and zero-point motion, which a static relaxation leaves out.

The singlet needs care. The a¹Δ<sub>g</sub> state of O<sub>2</sub> is not
a single closed-shell determinant, and OMol25 ran main-group singlets in
restricted Kohn-Sham unless bonds were expected to break
[[3](https://arxiv.org/abs/2505.08762)]. The OMol25 singlet is therefore
a closed-shell ωB97M-V state, not the experimental a¹Δ<sub>g</sub>, and
the reference method itself overestimates the 0.982 eV gap; a model
trained on it should too.

## Results

One MI250X GCD on `dev-g`, torch 2.7.1+rocm6.2.4, float64; orb-models
0.5.5 and fairchem-core 2.21.0. Each job (both checks and both timing
runs) took about 4 min: 3 min 39 s for Orb, 3 min 58 s for UMA.

:::{dropdown} results (CSV and JSON)
```{literalinclude} ../examples/uma/results/orb/lattice.csv
:language: text
```
```{literalinclude} ../examples/uma/results/uma/lattice.csv
:language: text
```
```{literalinclude} ../examples/uma/results/orb/molecules.json
:language: json
```
```{literalinclude} ../examples/uma/results/uma/molecules.json
:language: json
```
```{literalinclude} ../examples/uma/results/orb/timing_float64.csv
:language: text
```
```{literalinclude} ../examples/uma/results/uma/timing_float64.csv
:language: text
```
:::

### Crystals

| Crystal | Orb-v3 (Å) | UMA `omat` (Å) | PBE (Å) | Experiment (Å) |
|---|---|---|---|---|
| Si | 5.4603 (−0.18 % vs PBE) | 5.4607 (−0.17 %) | 5.470 | 5.431 |
| Cu | 3.6134 (−0.43 %) | 3.6154 (−0.37 %) | 3.629 | 3.615 |

Both models land within 0.5 % of PBE, slightly below it, and within
0.002 Å of each other. Cu then matches experiment, but only because the
model error and the PBE overestimate cancel; Si keeps most of the PBE
error (+0.5 % against experiment).

### Molecules

| Quantity | OrbMol | UMA `omol` | Reference |
|---|---|---|---|
| O<sub>2</sub> singlet minus triplet (eV) | 1.544 | 1.504 | 0.982 (experiment, a¹Δ<sub>g</sub>) |
| O<sub>2</sub> bond, triplet / singlet (Å) | 1.188 / 1.196 | 1.197 / 1.196 | |
| O<sub>2</sub> ionisation energy, vertical (eV) | 12.66 | 12.71 | 12.07 (experiment, adiabatic) |
| H<sub>2</sub>O ionisation energy, vertical (eV) | 6.07 | 12.64 | 12.62 (experiment) |
| H<sub>2</sub>O ionisation energy, adiabatic (eV) | 6.06 | 12.55 | 12.621 (experiment) |

- O<sub>2</sub> spin: both models put the singlet 1.5 eV above the
  triplet, 0.5 eV more than experiment, as expected from a closed-shell
  singlet reference. The two agree to 0.04 eV, which points at the
  reference method rather than the models.
- O<sub>2</sub> charge: both vertical ionisation energies are about
  0.6 eV above the adiabatic experimental value, a plausible size for a
  vertical value.
- Water: UMA is within 0.02 eV of the experimental vertical value, and
  its cation relaxes (O–H 0.962 to 0.999 Å), lowering the energy by
  0.08 eV. OrbMol gives half the ionisation energy and a cation that
  barely moves. The charge input reaches OrbMol (O<sub>2</sub> is
  right), so this checkpoint does not describe the water cation.
  OrbMol-v2 adds explicit electrostatics aimed at charge states like this
  but needs a newer software stack than the CSC module.

A model that takes charge and spin as inputs will return a number for any
combination. Check each charge and spin state against a known reference
before using it.

### Cost on one MI250X GCD

| System | Orb load / first / per call (s) | UMA load / first / per call (s) |
|---|---|---|
| Si, 216 atoms | 8.8 / 1.18 / 0.058 | 22.4 / 1.82 / 0.137 |
| Cu, 256 atoms | 8.8 / 0.10 / 0.097 | 22.4 / 0.25 / 0.248 |
| H<sub>2</sub>O, 3 atoms | 1.1 / 0.027 / 0.022 | 22.4 / 0.078 / 0.061 |

- Orb loads one checkpoint per task (Orb-v3 9 s, OrbMol 1 s); UMA loads
  one for all tasks (22 s). These are warm loads; the first load in a
  fresh job step took about 30 s for Orb-v3 and 68 s for UMA.
- Per force call, Orb-v3 is 2.3 to 2.6 times faster than UMA on these
  crystals.
- The first call includes a one-off GPU warm-up, paid once per process
  (Si ran first; Cu reuses it).
- Three atoms cost 20 to 60 ms per call: small molecules are bound by
  kernel launch and Python overhead, not by arithmetic.
- In float32 the crystal calls fail with `CUDA driver error: 209` (the
  cell determinant) for both models. OrbMol on the non-periodic molecule
  runs in float32 at 21 ms per call; for UMA the water call failed too,
  after the crystal failure in the same process.

:::{keypoints}
- Orb covers crystals and molecules with two checkpoints; UMA with one
  checkpoint and a task name. Either way, each model reproduces its own
  reference level: PBE for crystals, ωB97M-V for molecules.
- Charge and spin multiplicity are inputs in `atoms.info`; ASE caching
  ignores them, so reset the calculator between states.
- Crystal results agree closely between Orb-v3 and UMA. For molecules, a
  charge-aware model can still be wrong for one charge state: OrbMol
  halves the water ionisation energy, UMA reproduces it.
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
9. B. Rhodes et al., Orb-v3. [arXiv:2504.06231](https://arxiv.org/abs/2504.06231)
10. Orbital Materials, OrbMol: extending Orb to molecular systems (2025).
    [orbitalindustries.com](https://www.orbitalindustries.com/posts/orbmol-extending-orb-to-molecular-systems)
11. NIST Chemistry WebBook, O<sub>2</sub>, gas-phase ion energetics.
    [webbook.nist.gov](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=20)
