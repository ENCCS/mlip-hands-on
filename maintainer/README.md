# Maintainer checks

Participants use `content/`, `examples/`, and `scripts/`. This directory is
optional. `check-public-content.py` scans tracked text for known private
values and confirms Sphinx will not run GPU cells at publication time.

The older Python-LAMMPS drivers and site-specific jobs remain recoverable in
the `main` branch's Git history, not in this participant-facing tree. They
must not be used to benchmark this branch. Fresh GPU qualification invokes
the participant commands from a distinct stage, with exact model/runtime
hashes, account, partition, wall time, and result identity recorded privately.
