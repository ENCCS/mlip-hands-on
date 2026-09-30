# Arrhenius one-GPU MPI-rank capacity check

This is a bounded functional check, not a throughput benchmark or a claim of
the absolute capacity limit. The scheduler allocated one GH200 GPU on one
Arrhenius node. The model was MACE-MP-0a small, exported for native LAMMPS
ML-IAP/Kokkos. CUDA MPS was started in a private node-local directory.

Source identities:

- LAMMPS executable SHA-256: `2bab746b564b6d0dcacf86be8aad2e24b62c720c7a2a22e8dbec544c3d29b5d4`
- Exported model SHA-256: `db578c556298ad3bb1f4a93faa50d540eb2b9792215e81ef7548dd7e20a746e8`
- `examples/lammps_mace.in` SHA-256: `e40764e2bc7a3e6ec03c5d9dacd3b920663303990415efa9dad579cc7bc03e5e`
- Final shared-rank launcher SHA-256: `dce76ac6275773de5178856fca7d4836fec33bf93b5df15e360a0c4be82e2cc4`

All completed 39,304-atom runs used 10 warm-up and 2 further MD steps.
The one-rank result came from job `3178716`; equal-warm-up two- and four-rank
results came from completed job `3179045`. The latter job's allocated TRES
included exactly one GH200 GPU. The initial coarse sweep `3178446` showed
32,768 atoms completed at all three rank counts and CUDA OOM at 64,000 atoms.
Its 110,592-atom one-rank attempt also reported CUDA OOM. Job `3178986`
completed a separate four-rank 39,304-atom check before the equal-warm-up
confirmation.

In `3178716`, the 46,656-atom one- and two-rank attempts produced explicit
`torch.OutOfMemoryError` during the first force evaluation. The two-rank
step did not close promptly after one rank failed, so the exact job was
cancelled. The coarse sweep `3178446` was also cancelled after the
110,592-atom two-rank attempt reported OOM and remained in its Slurm step.
These cancellations do not invalidate earlier completed rows but are not
completed jobs or performance evidence. No uncertain submission was retried.

The highest completed size among the tested cubic cells was `cells=17`, or
39,304 atoms. `cells=18` is 46,656 atoms; it was not tested with four ranks.
Larger or differently shaped systems, other models, build options, and
longer trajectories may change the boundary. The two-step loop times must
not be quoted as sustained throughput.
