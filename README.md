# Universal MLIPs on HPC: hands-on

Source of the ENCCS lesson at <https://enccs.github.io/mlip-hands-on/>.

- Part A: universal MLIPs for screening and training (TorchSim, Orb, MatGL,
  fine-tuning, NEB, UMA), in `examples/`.
- Part B: silicon MD with MACE in NVIDIA ALCHEMI Toolkit and LAMMPS.
- Webinar slides: the `slides` page.

Every page shows saved results, so it reads without a GPU. Running the
examples needs a GPU system; see `setup/`. UMA weights are gated (accept the
licence on Hugging Face and use a token).

## Build

```bash
uv run --with-requirements requirements.txt make html
# live preview: uv run --with-requirements requirements.txt make livehtml PORT=8766
```

Pushes to `main` are built and published by GitHub Actions. Model weights,
containers and credentials are never stored in Git.

## Licence

Text and pedagogical material: CC BY-SA 4.0 (`LICENSE`). Code: MIT
(`LICENSE.code`). Reused figures keep the licence in their caption.

## Acknowledgements

<img src="_static/EN_Co-fundedbytheEU_RGB_POS.png" alt="Co-funded by the European Union" height="60">

EuroCC 3 has received funding from the European High-Performance Computing Joint Undertaking (JU) under Grant Agreement No. 101306701. The JU receives support from the European Union‘s Digital Europe Programme and Germany, Albania, Austria, Belgium, Bosnia and Herzegovina, Bulgaria, Croatia, Cyprus, Czechia, Denmark, Estonia, Finland, France, Greece, Hungary, Iceland, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Montenegro, the Netherlands, North Macedonia, Norway, Poland, Portugal, Romania, Serbia, Slovakia, Slovenia, Spain, Sweden, Türkiye, and Kosovo.

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or EuroHPC Joint Undertaking. Neither the European Union nor the EuroHPC Joint Undertaking can be held responsible for them.

ENCCS has also received national funding through Vinnova and the Swedish Research Council (VR).

The project is supported by the European High-Performance Computing Joint Undertaking and its members.
