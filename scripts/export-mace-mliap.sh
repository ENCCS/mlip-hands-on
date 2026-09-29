#!/usr/bin/env bash
# MACE writes <checkpoint>-mliap_lammps.pt beside the checkpoint.
test ! -e "$1-mliap_lammps.pt" || { echo 'export already exists' >&2; exit 2; }
exec python -m mace.cli.create_lammps_model "$1" --format mliap --dtype float32
