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

The [Python source](../../examples/alchemi_si_one_cell.py) shown in the
previous episode changes only `MLIP_REPLICAS`: the loop creates independently
seeded `AtomicData` objects, then `Batch.from_data_list` collects them for
one model evaluation. These cells invoke that same source inside the SIF;
they do not implement a second MD program.

```{code-cell} ipython3
%%bash
cd ../..
time apptainer exec --cleanenv --nv \
  --env MLIP_REPLICAS=1 --env MLIP_STEPS=2000 \
  --env TORCH_COMPILE_DISABLE=1 --env TORCH_DISABLE_NATIVE_JIT=1 \
  --bind "$MLIP_MODEL:/models/mace.model:ro" \
  --bind "$PWD/examples/alchemi_si_one_cell.py:/opt/mlip/md.py:ro" \
  "$MLIP_ALCHEMI_SIF" python /opt/mlip/md.py
```

```{code-cell} ipython3
%%bash
cd ../..
time apptainer exec --cleanenv --nv \
  --env MLIP_REPLICAS=8 --env MLIP_STEPS=2000 \
  --env TORCH_COMPILE_DISABLE=1 --env TORCH_DISABLE_NATIVE_JIT=1 \
  --bind "$MLIP_MODEL:/models/mace.model:ro" \
  --bind "$PWD/examples/alchemi_si_one_cell.py:/opt/mlip/md.py:ro" \
  "$MLIP_ALCHEMI_SIF" python /opt/mlip/md.py
```

For a throughput comparison, count *all* completed replica steps and divide
by elapsed seconds. Eight replicas take more total work than one; a shorter
time per replica does not mean the eight-replica job finishes sooner.

```{note}
The example selects Langevin NVT. Its source shows where to select a
Nose–Hoover thermostat instead. Thermostat details differ from LAMMPS;
compare performance for a stated setup, not individual stochastic paths.
```
