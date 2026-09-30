# Maintainer checks

Participants use `content/`, `examples/`, and `scripts/`. This directory is
optional. `check-public-content.py` scans tracked text for known private
values and confirms Sphinx will not run GPU cells at publication time.
`check-notebook-source.py` proves that the visible one-trajectory Python cell
matches the CLI source exactly. `test-notebook-alchemi.py` checks that its
container adapter sends the visible source to Python without a shell or an
implicit second simulation.
`qualify-notebook.py` executes the actual MyST page inside a separately
approved site allocation and reports a bounded outcome. On Arrhenius, page 06
explicitly reports its eight-process LAMMPS cell as skipped; it must not be
counted as an eight-replica qualification.

The bounded Jupyter site jobs and laptop connector live in `scripts/` because
participants need them. Other older scientific site jobs remain recoverable
in the `main` branch's Git history; they must not be used to benchmark this
branch. Fresh GPU qualification invokes the participant commands from a
distinct stage, with exact model/runtime hashes, account, partition, wall
time, and result identity recorded privately.
