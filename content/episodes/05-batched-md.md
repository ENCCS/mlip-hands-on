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

# Run independent trajectories together

The [ALCHEMI example](../../examples/alchemi_si_one_cell.py) builds one
64-atom silicon structure, then makes eight independent copies. Each copy
gets a different initial velocity seed. `Batch.from_data_list` combines
their atom data so the MACE model evaluates them in one batched operation;
the trajectories do not exchange atoms or forces.

On an allocated GPU, run one replica and then eight. Keep the model, step
count, and GPU the same. The printed energies are physical outputs, not
timings; use a clock around the commands when measuring throughput.

```{code-cell} ipython3
%%bash
cd ../..
time bash scripts/run-alchemi.sh "$PWD" "$MLIP_ALCHEMI_SIF" "$MLIP_MODEL" 1 2000
```

```{code-cell} ipython3
%%bash
cd ../..
time bash scripts/run-alchemi.sh "$PWD" "$MLIP_ALCHEMI_SIF" "$MLIP_MODEL" 8 2000
```

For a throughput comparison, count *all* completed replica steps and divide
by elapsed seconds. Eight replicas take more total work than one; a shorter
time per replica does not mean the eight-replica job finishes sooner.

```{note}
The example selects Langevin NVT. Its source shows where to select a
Nose–Hoover thermostat instead. Thermostat details differ from LAMMPS;
compare performance for a stated setup, not individual stochastic paths.
```
