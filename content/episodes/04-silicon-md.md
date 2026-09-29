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

# Run one silicon trajectory

Start with 64 silicon atoms on one allocated GPU. The ALCHEMI program
[`alchemi_si_one_cell.py`](../../examples/alchemi_si_one_cell.py) creates
the diamond cell, loads the original MACE checkpoint, and runs Langevin MD.
The LAMMPS [input file](../../examples/lammps_mace.in) uses the exported
model with NVE integration and a Langevin thermostat. Both use a 0.1 fs
timestep, but their thermostat implementations need not yield identical
trajectories.

This notebook cell runs the ALCHEMI example:

```{code-cell} ipython3
%%bash
cd ../..
bash scripts/run-alchemi.sh "$PWD" "$MLIP_ALCHEMI_SIF" "$MLIP_MODEL" 1 200
```

It prints a completion line and final potential energy. Increase the step
count to 2,000 to see its 1,000-step energy/temperature progress lines.

The LAMMPS calculation uses its CLI and `.in` file:

```{code-cell} ipython3
%%bash
cd ../..
bash scripts/run-lammps.sh "$PWD" "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 2 200
```

LAMMPS prints step, atom count, temperature, and potential energy. The
input currently prints every ten steps; it includes a commented 1,000-step
option for longer runs. The next episode repeats independent ALCHEMI
trajectories on one GPU.
