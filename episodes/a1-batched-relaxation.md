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

# Batched relaxation for screening

Screening relaxes many small, independent structures. This page relaxes one
set twice with the same MACE-MP foundation model: serially with ASE, then in
one batched [TorchSim](https://github.com/TorchSim/torch-sim) call on one
GPU.

Part A has its own pixi environment and uses MACE-MP-0b small, not the
MACE-MP-0a checkpoint pinned for Part B.

The workload is a toy screening set: rattled copies of four crystals
(Cu fcc, Si diamond, Fe bcc and Al fcc, 8 to 32 atoms each). Each copy has a
different random displacement and needs its own relaxation.

```{literalinclude} ../examples/torchsim/workload.py
:language: python
:start-at: FAMILIES
:lineno-match:
```

## Serial and batched runs

Both routes use FIRE with a fixed cell and stop when the largest force falls
below `--fmax` (0.05 eV/Å by default) or after `--max-steps`. The serial
baseline relaxes only the first `--baseline-n` structures and extrapolates
its per-structure time to the full set; a full serial run would use most
of the allocation.

```{literalinclude} ../examples/torchsim/runners.py
:language: python
:start-at: class SerialRelaxer
:lineno-match:
```

TorchSim takes the whole list of ASE `Atoms`, builds one batched state and
advances every structure together. Converged structures leave the batch;
with `--autobatch` TorchSim also splits the set to fit GPU memory. Both
routes use the same checkpoint. TorchSim needs the raw PyTorch model, so it
is loaded with `return_raw_model=True`:

```{literalinclude} ../examples/torchsim/models.py
:language: python
:start-at: def load_mace
:end-before: def load_models
:lineno-match:
```

After both runs, the example compares the ASE and TorchSim energies on the
structures relaxed both ways and writes `summary.json` and
`relaxed.extxyz`. When `--checkpoint` names a local file, the summary also
records its SHA-256.

## Laptop run

The example is a Python package in `examples/torchsim/` with its own
[pixi](https://pixi.sh) environment; its entry point is
[`examples/torchsim/__main__.py`](../examples/torchsim/__main__.py). The
`smoke` task uses a toy Lennard-Jones potential. It needs no download and
finishes in seconds; its timings and energies have no physical meaning.

```bash
cd examples/torchsim
pixi run smoke
```

The task writes to `examples/torchsim_smoke/`. The example refuses to
overwrite an existing output directory, so remove it or pass a new
`--outdir` before running again.

Expected output (numbers vary):

```text
model=lj device=cpu dtype=float64 structures=8 serial subset=2
                             wall time   s/structure
serial, measured                 0.1 s        0.03 s   (2 structures)
serial, estimated                0.2 s        0.03 s   (8 structures)
batched, measured                5.4 s        0.68 s   (8 structures)
estimated serial / batched: 0.04
max |E_ASE - E_TorchSim| on the serial subset: 2.03e-01 eV
```

On a CPU with a cheap potential, batching is slower because there is no
GPU parallelism. The energy difference arises because ASE and TorchSim use
different cutoff and energy-shift conventions for Lennard-Jones, so the two
toy potentials differ. Only the MACE run is a consistency check (a few meV).

To run the MACE model on a CPU, use the `cpu` task. It relaxes four
structures for at most 20 steps and writes to `examples/torchsim_cpu/`. The
first run downloads the MACE-MP-0b small checkpoint, about 68 MB:

```bash
pixi run cpu
```

## Leonardo run (one A100)

Compute nodes have no internet access. On a login node, install the pixi
environment inside your lesson copy and download the checkpoint to a
private location outside Git:

```bash
cd /leonardo_scratch/fast/<PROJECT>/<USER>/mlip-md-lesson/examples/torchsim
pixi install
curl -L -o <SCRATCH>/models/mace-mp-0b-small.model \
  https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model
printf '%s  %s\n' \
  7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0 \
  <SCRATCH>/models/mace-mp-0b-small.model | sha256sum -c -
```

The job checks this SHA-256 before it starts and refuses any other file.

Then submit the one-GPU job from the repository root. It refuses missing
inputs and an existing output directory:

```bash
export MLIP_LESSON_ROOT=/leonardo_scratch/fast/<PROJECT>/<USER>/mlip-md-lesson
export MLIP_TORCHSIM_CHECKPOINT=<SCRATCH>/models/mace-mp-0b-small.model
export MLIP_RESULTS_DIR=/path/to/private/results
sbatch --account=<PROJECT> \
  --export=ALL,MLIP_LESSON_ROOT,MLIP_TORCHSIM_CHECKPOINT,MLIP_RESULTS_DIR \
  scripts/test-leonardo-torchsim.sbatch
```

```{literalinclude} ../scripts/test-leonardo-torchsim.sbatch
:language: bash
:start-at: outdir=
:lineno-match:
```

This script has **not yet been qualified**. The float64 row below comes
from an earlier script with the same workload and options; the float32 row
comes from a separate tuned run with the options listed below the table.
Neither was produced by this script.

## Leonardo results

One A100, MACE-MP-0b small, no cuEquivariance kernels. The serial column
measured 8 structures; the estimate scales that time to the full set.

| Precision and batching | Structures | Serial, 8 measured (s) | Serial, estimated (s) | Batched (s) | Estimated serial / batched |
|---|---:|---:|---:|---:|---:|
| float64 | 64 | 37.4 | 299.5 | 59.3 | 5.05 |
| float32, `--autobatch` | 128 | 38.7 | 619.6 | 85.2 | 7.27 |

The float32 run used `--dtype float32 --n-variants 32 --autobatch`.

:::{warning}
These are single examples, not a benchmark.

- Each row is **one run**. There are no repeats, medians or ranges.
- The serial time for the full set is **extrapolated** from 8 structures,
  not measured.
- The model is **MACE-MP-0b small**. It is not the MACE-MP-0a checkpoint
  pinned for Part B, so the energies and costs are not interchangeable.
- The runs used no cuEquivariance kernels; they would change the timings.
- This is **relaxation, not molecular dynamics**. Do not compare these
  times with the Part B MD timings.
:::
