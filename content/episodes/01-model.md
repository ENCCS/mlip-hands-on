# One MACE potential, two engines

MACE is a machine-learned interatomic potential: it predicts energy and
atomic forces from a structure. The examples use one pinned MACE-MP-0a
small checkpoint. Download it to private storage and set `MLIP_MODEL`
to its path. [The model record](../reference/model.toml) lists its source
and SHA-256; keep the weights outside Git.

ALCHEMI reads the trusted original checkpoint. LAMMPS ML-IAP reads an
exported `.pt` file from the same checkpoint:

```bash
bash scripts/export-mace-mliap.sh "$MLIP_MODEL"
```

This writes a second file beside the checkpoint. Set `MLIP_MLIAP_MODEL`
to that output, not to the original checkpoint. The script runs
`python -m mace.cli.create_lammps_model "$MLIP_MODEL" --format mliap --dtype float32`
and refuses to overwrite an existing export.

```{note}
The files represent the same model in different formats: ALCHEMI needs the
original checkpoint; LAMMPS ML-IAP needs the export. Do not swap their paths.
```

The silicon examples use periodic diamond Si at lattice parameter 5.43 Å.
This is a small, readable workload, not a validation of the model for all
silicon conditions.
