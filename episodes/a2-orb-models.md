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

# Orb-v3, MatGL and dispersion

:::{objectives}
- Run the screening workflow with a different universal model.
- Test whether a model trained on PBE data binds graphite layers.
- Add a D3 dispersion correction and judge the result against the right
  reference.
:::

Two questions follow from batched screening: does the workflow depend on
one model, and what do models trained on PBE data miss? PBE contains no
dispersion (see {ref}`background-foundation`), and layered materials are
held together by it. The Part A workflow is not specific to MACE. This
page runs the same screening with an Orb-v3 model and a MatGL TensorNet
model, then uses graphite, a layered crystal, to show what a universal MLIP
trained on PBE data misses: the van der Waals (dispersion) attraction
between layers.

## Orb-v3

[Orb-v3](https://arxiv.org/abs/2504.06231) (Rhodes et al., 2025) is a
family of universal potentials from Orbital Materials, released under the
Apache-2.0 licence in the
[`orb-models`](https://github.com/orbital-materials/orb-models) package.
Unlike MACE, it is not equivariant by construction; the authors report
accurate physical properties at much lower latency and memory use. For
where Orb-v3 sits among other fast universal models, see
{ref}`background-choosing`. The model names encode three choices:

| Part of the name | Options | Meaning |
|---|---|---|
| `conservative` / `direct` | forces from the energy gradient, or predicted directly | conservative is slower but gives a consistent energy surface; use it for relaxations, elastic constants, phonons and NVE MD |
| `inf` / `20` | up to 120 neighbours, or at most 20 | the cap of 20 makes the energy surface discontinuous |
| `omat` / `mpa` | trained on OMat24, or on MPtrj and Alexandria | the Orb authors advise `omat` unless a benchmark needs `mpa` |

The example uses `orb-v3-conservative-inf-omat`. Its training data are PBE
and PBE+U calculations without a dispersion correction, so the model has
learnt no dispersion, regardless of its 6 Å cutoff.

## Switch the model

The Orb loader returns a model and an atoms adapter. The same objects feed
the ASE calculator for the serial baseline and the TorchSim wrapper for the
batch:

:::{dropdown} Code: models.py
```{literalinclude} ../examples/torchsim/models.py
:language: python
:start-at: def load_orb
:end-before: def load_models
:lineno-match:
```
:::

To repeat the laptop CPU check of the previous page with Orb, change
only `--model`. The first run downloads the checkpoint, about 100 MB:

```bash
cd examples
pixi run --manifest-path torchsim/pixi.toml \
  python -m torchsim --model orb-v3-conservative-inf-omat --device cpu \
  --n-variants 1 --baseline-n 1 --max-steps 20 --outdir orb_cpu
```

On a laptop CPU, the ASE and TorchSim energies of the shared structure
differed by 4 × 10⁻⁴ eV after at most 20 relaxation steps. This reflects
the two optimiser paths, not the model: a single-point energy of the same
structure agrees to about 10⁻¹² eV.

Three implementation details affect accuracy. Set the precision through the loader
(`precision="float32-highest"` or `"float64"`), not with
`torch.set_float32_matmul_precision` beforehand: the loader resets it.
The Orb and MACE loaders also change torch's global default dtype, so the
example restores it after each load; otherwise the results can depend on
the order in which models are loaded. Finally, the Orb ASE calculator
builds its input graph in that global default dtype, so the example sets
the requested dtype for each call; without this, a float64 model receives
float32 positions. The example also turns off `torch.compile`
(`compile=False`), which the loader otherwise applies on the first call
and so adds a one-off cost to the timed serial baseline.

## MatGL

[MatGL](https://github.com/materialyzeai/matgl) (BSD-3-Clause) provides
TensorNet, CHGNet, M3GNet, QET and other graph models, with pretrained
potentials trained on MatPES (PBE or r2SCAN) and distributed through
Hugging Face. Since version 4.0 it runs on PyTorch Geometric only; the DGL
backend has been removed. The example uses
`TensorNet-PES-MatPES-PBE-2025.2` (0.84 M parameters, 5 Å cutoff) through
MatGL's ASE calculator:

:::{dropdown} Code: models.py
```{literalinclude} ../examples/torchsim/models.py
:language: python
:start-at: def load_matgl
:end-before: def load_models
:lineno-match:
```
:::

TorchSim 0.6 has no MatGL model interface, so this model runs the serial
ASE baseline only. The pretrained models also compute in float32. The
first run downloads the model to the MatGL cache (`MATGL_CACHE`):

```bash
cd examples
pixi run --manifest-path torchsim/pixi.toml \
  python -m torchsim --model matgl-tensornet-pbe --device cpu --dtype float32 \
  --n-variants 1 --baseline-n 1 --max-steps 20 --outdir matgl_cpu
```

Output on a laptop CPU (`matgl` 4.0.3, `torch` 2.9.1):

```text
model=matgl-tensornet-pbe device=cpu dtype=float32 structures=4 serial subset=1
matgl-tensornet-pbe has no TorchSim interface; ran the ASE baseline only
serial, measured: 0.9 s, 0.86 s/structure (1 structures)
```

The relaxed 32-atom copper cell has an energy of −119.344 eV. Set
`stress_unit="eV/A3"` on `PESCalculator`, as the loader does: its default
is GPa, whereas ASE expects eV/Å³.

## Graphite interlayer spacing

![Graphite test: model variants, cell relaxation, comparison with PBE and experiment.](../_static/graphite-test.drawio.png)

Bernal (AB) graphite has four atoms per cell. Its layers are held together
almost entirely by dispersion. Plain PBE gives an interlayer spacing of
4.40 Å and a binding energy of only 1 meV per carbon atom
([Hazrati et al., 2014](https://doi.org/10.1103/PhysRevB.90.155448)),
against 3.34 Å in experiment
([Baskin and Meyer, 1955](https://doi.org/10.1103/PhysRev.100.544), as
tabulated by Hazrati et al.). A model trained on PBE data inherits this
error, regardless of its cutoff; dispersion must come from an added term
such as D3.

[`examples/torchsim/layered.py`](../examples/torchsim/layered.py) relaxes the
cell and positions with ASE (`FrechetCellFilter` and FIRE, 0.002 eV/Å) for
three variants by default: MACE-MP-0b small, Orb-v3, and Orb-v3 plus
Grimme's D3(BJ) correction with PBE parameters. `--variants tensornet` runs
the MatGL model instead, in float32. The D3 term comes from `orb-models` itself:

```python
orbff = D3SumModel(orbff, AlchemiDFTD3(functional="PBE", damping="BJ"))
```

PBE is the correct D3 functional here because it matches the training data.
Do not add D3 to a model already trained with a dispersion correction, such
as `orb-d3-v2`; that counts dispersion twice. This page adds D3 to Orb only.

Each model relaxes twice: from the experimental spacing (3.34 Å) and from
the plain-PBE spacing (4.4 Å). If the two results differ, the energy surface
is too flat to define a spacing. The study writes one row per run to
`graphite.json`, with package versions, step counts and whether it
converged:

```bash
cd examples/torchsim
pixi run graphite-cpu
```

Output on a laptop CPU (one run, float64, `orb-models` 0.7.0):

```text
variant      start d   a (A)   d (A)   d err  steps  converged
mace-small      3.34   2.467   4.101   22.8%     58  True
mace-small      4.40   2.467   4.099   22.7%     34  True
orb-v3          3.34   2.468   4.310   29.0%    526  True
orb-v3          4.40   2.468   4.365   30.7%    135  True
orb-v3+d3       3.34   2.466   3.443    3.1%    103  True
orb-v3+d3       4.40   2.466   3.448    3.2%    229  True
tensornet       3.34   2.464   4.884   46.2%    588  True
tensornet       4.40   2.464   4.883   46.2%    429  True
PBE                    2.470   4.400
experiment             2.460   3.340
```

Here `d = c/2` is the interlayer spacing and `d err` is measured against
experiment. Both models give the in-plane lattice constant `a` within
0.4 % of experiment. Without D3 the reference is PBE, not experiment:
plain Orb-v3 (4.31 to 4.37 Å) is close to the PBE value of 4.40 Å, while
MACE stops shorter, at 4.10 Å. The smaller MACE error therefore does not
indicate better interlayer binding; on a nearly unbound surface, small
fitting differences shift the spacing considerably. The plain Orb-v3
spacing is also poorly defined: the two starts end 0.05 Å apart after more
than 500 steps, which indicates a nearly flat interlayer energy surface.
With D3, Orb-v3 has a clear minimum within about 3 % of experiment, and
both starts agree to 0.005 Å. With the looser threshold of 0.01 eV/Å, the
D3 starts differed by about 0.04 Å, so check convergence before quoting a
spacing. Step counts on this flat surface can change by a few between
runs; the spacings do not.

The TensorNet row (`--variants tensornet`, `matgl` 4.0.3, float32) shows
the effect of a short cutoff. Its energy falls monotonically as the layers
separate, by 36 meV per atom from 3.34 to 4.88 Å, and is constant beyond
about 5 Å, the model cutoff. Both starts therefore stop at 4.88 Å, where
the interlayer force vanishes; the model has no interlayer minimum at all.

:::{note}
The experimental value is a low-temperature measurement; the relaxations
are static and include no zero-point motion or temperature. In
`orb-models` 0.7.0 the D3 neighbour list assumes atoms inside the cell;
the graphite cell is built that way and the atoms do not move out of it. A fix for unwrapped positions exists only on the main
branch.
:::

## Leonardo run (one A100)

Set the variables of the previous page. On a login node, update the pixi
environment of your lesson copy (this page adds `orb-models`, and compute
nodes have no internet), then download the Orb checkpoint next to the MACE
one and check its SHA-256. Then submit the job, which runs the graphite
study and the Orb screening batch:

:::{dropdown} Commands
```bash
cd "$MLIP_LESSON_ROOT/examples/torchsim"
pixi install
cd "$MLIP_LESSON_ROOT"
curl -L -o <SCRATCH>/models/orb-v3-conservative-inf-omat.ckpt \
  https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt
printf '%s  %s\n' \
  0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0 \
  <SCRATCH>/models/orb-v3-conservative-inf-omat.ckpt | sha256sum -c -
export MLIP_ORB_CHECKPOINT=<SCRATCH>/models/orb-v3-conservative-inf-omat.ckpt
sbatch --account=<PROJECT> \
  --export=ALL,MLIP_LESSON_ROOT,MLIP_TORCHSIM_CHECKPOINT,MLIP_ORB_CHECKPOINT,MLIP_RESULTS_DIR \
  scripts/test-leonardo-orb.sbatch
```
:::

This script has **not yet been qualified**. The laptop results above are
the reference for this page.

## Reading the results

Changing the model needs only the `--model` option (and float32 for
TensorNet). For graphite, no model without D3 reproduces the experimental
spacing: plain Orb-v3 stays near the PBE value of 4.40 Å, and TensorNet,
with its 5 Å cutoff, has no interlayer minimum at all. With D3, Orb-v3
lands within about 3 % of experiment, so judge an uncorrected model
against PBE, not experiment.

:::{keypoints}
- The batched workflow accepts other universal models; TensorNet runs the
  serial ASE route only.
- Models trained on PBE data miss the dispersion that binds graphite layers.
- D3 adds dispersion; do not add it to a model already trained with it.
- Start from two spacings and check convergence before quoting a result.
:::

## References

- B. Rhodes et al., *Orb-v3: atomistic simulation at scale*,
  [arXiv:2504.06231](https://arxiv.org/abs/2504.06231) (2025).
- Y. Baskin and L. Meyer, *Lattice constants of graphite at low
  temperatures*, [Phys. Rev. 100, 544](https://doi.org/10.1103/PhysRev.100.544)
  (1955).
- E. Hazrati, G. A. de Wijs and G. Brocks, *Li intercalation in graphite: a
  van der Waals density-functional study*,
  [Phys. Rev. B 90, 155448](https://doi.org/10.1103/PhysRevB.90.155448) (2014).
- S. Grimme, S. Ehrlich and L. Goerigk, *Effect of the damping function in
  dispersion corrected density functional theory*,
  [J. Comput. Chem. 32, 1456](https://doi.org/10.1002/jcc.21759) (2011).
