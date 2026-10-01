# The two GH200 systems

Both sites pair an Arm Grace CPU closely with a Hopper GPU. A compute node
contains four GH200 superchips. The first examples use one GPU; the site's
scheduler determines how much of the node you must allocate. The
GPU's HBM and the Grace CPU's memory are different memory pools; a program
using the GPU does not automatically gain the sum as GPU memory.

| Site | One compute node | GPU memory | CPU-side memory per superchip | Fabric |
| --- | --- | --- | --- | --- |
| Arrhenius GPU | 4 Grace CPUs + 4 Hopper GPUs | 96 GB HBM per GPU | 128 GB LPDDR | Slingshot |
| JUPITER Booster | 4 Grace CPUs + 4 Hopper GPUs | 96 GB HBM per GPU | 120 GB LPDDR5X | InfiniBand |

These are [NAISS's Arrhenius specifications](https://www.naiss.se/resources/arrhenius-technical-description/)
and [JSC's JUPITER specifications](https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html).
Use `nvidia-smi` in your allocation to check the visible GPUs and memory.

```{figure} ../_static/jupiter-booster-racks.jpg
:alt: JUPITER Booster racks
:width: 70%

JUPITER Booster racks. Image credit: Forschungszentrum Jülich / Sascha Kreklau;
license and source are recorded in `THIRD_PARTY.md` in the lesson repository.
```

The scaling episode uses several GPUs for one LAMMPS trajectory. Follow
your site's setup page: the MPI stack and Slurm options differ.
