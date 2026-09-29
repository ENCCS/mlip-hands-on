# One MACE potential, two engines

MACE is a machine-learned interatomic potential: it predicts energy and
atomic forces from a structure. The examples use one pinned MACE-MP-0a
small checkpoint. Its source and SHA-256 are in
[the model record](../reference/model.toml). Download the original model
to private storage; do not commit its weights.

ALCHEMI reads the trusted original checkpoint. LAMMPS ML-IAP reads an
exported `.pt` file from the same checkpoint:

```bash
bash scripts/export-mace-mliap.sh "$MLIP_MODEL"
```

The exporter writes a second artifact beside the checkpoint. Point
`MLIP_MLIAP_MODEL` to the output. The files are different formats; neither
should be substituted for the other. Before a class, check that the export
loads with the pinned LAMMPS build.

The silicon examples use periodic diamond Si at lattice parameter 5.43 Å.
This is a small, readable workload, not a validation of the model for all
silicon conditions.
