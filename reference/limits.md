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

# Limits and provenance

The pinned silicon/MACE runs show how to execute and measure MD.
They do not establish that this model is accurate for a new scientific
question. Validate the model against suitable reference data before drawing
materials conclusions.

The checked-in `reviewed-shared-gpu-nve.csv` contains two completed
one-GH200, eight-replica matrices: 64 and 512 atoms per replica. Each row
used 10 warmup and 200 measured NVE steps at 0.1 fs. The engines used
independently initialized velocities; the runs are not atom-by-atom matched.
Group wall includes startup, model loading, warmup, and MD, but not queue
wait or native archive extraction. A larger 4,096-atom matrix was incomplete
and is not included in that comparison.

The checked-in `reviewed-shared-gpu-nve-jupiter.csv` holds the same
eight-replica matrix, 64 and 512 atoms per replica, on one JUPITER GH200.
It is one run per case with different runtime packaging. At 64 atoms,
LAMMPS with CUDA MPS (73 s) finished before the ALCHEMI batch (76 s),
whose whole-workflow time there is dominated by startup; compare the two
sites only with that caveat.

A distinct one-GPU capacity probe completed a short ten-step workload with
six replicas of 8,000 atoms; seven replicas produced CUDA out-of-memory.
That brackets **that workload**, not a general atom limit. PyTorch-reported
peak allocation excludes native and driver allocations. The SIF, model,
neighbor list, precision, and competing GPU work change the usable size.

The separate ALCHEMI spatial-decomposition route has an unresolved
multi-GPU rank-count energy difference. One-node functional runs must not
be turned into a multi-GPU scientific-equivalence or two-node scaling claim.
LAMMPS 1/2/4-GPU timings are an application measurement, not proof that
GPU-direct MPI alone caused any speedup.

On JUPITER, the native ML-IAP/Kokkos route passed short 1/2/4-GPU one-node
and 8-GPU two-node functional tests with a pinned export. The MPI checks ran
only ten measured steps of one 512-atom system. Their final energies differed
slightly by rank count. They establish an executable route, not a throughput
ranking, a convergence result, or agreement with ALCHEMI. A separate native
ALCHEMI one-GPU check completed both one and eight independent 64-atom,
five-measured-step NVE trajectories; neither is a cross-engine validation or a
performance measurement.

The separate JUPITER Metatomic/Kokkos fork also completed five-step silicon
functional checks on one GPU, on two and four GPUs in one node, and on eight
GPUs in two nodes. The multi-GPU input had 512 atoms. A freshly exported
Metatomic model passed a one-GPU check, but its file hash differed from an
earlier export; record and verify each export separately. These checks add a
working path, not a speed comparison with ML-IAP or ALCHEMI.

The Part A batched-relaxation numbers are single runs on one Leonardo A100
with MACE-MP-0b small, not the MACE-MP-0a checkpoint pinned for Part B. The
serial totals are extrapolated from eight relaxed structures, no
cuEquivariance kernels were used, and the lesson's
`scripts/test-leonardo-torchsim.sbatch` has not yet been qualified. They
illustrate batched screening throughput, are not a benchmark and are not
comparable with the MD timings.

The Orb-v3 graphite spacings on the Part A pages are single static
relaxations from two starting spacings, compared with a low-temperature
experiment. `scripts/test-leonardo-orb.sbatch` has not yet been qualified.

Treat the plotted numbers as reviewed *examples* and label the configuration
and repetitions behind every performance statement. For a fresh benchmark,
predeclare the atom-count and replica matrix, repeat completed jobs, report
median and range, and keep single-system speed separate from aggregate
independent-replica throughput.
