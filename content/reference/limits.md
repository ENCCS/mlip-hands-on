# Interpreting the examples

A short completed run checks that the model, runtime, GPU and input work
together. Before using a result for research, check model suitability,
equilibration and longer-run stability separately.

ALCHEMI batches independent systems inside one calculation. Several LAMMPS
processes can share a GPU, but each keeps its own model and GPU context.
For one trajectory, report MD steps/s. For independent trajectories,
report total completed replica-steps/s and elapsed time. See the
[worked example](../episodes/09-results.md).

ALCHEMI FIRE and LAMMPS minimization can start from the same coordinates and
model, but their stopping rules differ. The relaxation page reports energies
and forces without assuming identical final states.

Matching performance inputs is not a physics validation. The known
large-system energy difference, timing variability and test coverage are
recorded in [Benchmark methods and checks](benchmarks.md).
