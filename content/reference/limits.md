# What the examples establish

A completed short run shows that a selected model, runtime, GPU, and input
work together. It does not establish long-time stability, physical accuracy,
temperature equilibration, numerical identity between MD engines, or a
production performance ranking.

ALCHEMI batches independent systems inside one calculation. Several LAMMPS
processes can share a GPU, but that is not the same operation. State both
the number of completed trajectories and the wall time for a declared
workload. Do not call one-trajectory speed aggregate throughput.

ALCHEMI FIRE and LAMMPS minimization can start from the same coordinates and
model, but their stopping rules differ. The relaxation page reports energies
and forces as observations, not a scientific-equivalence claim.
