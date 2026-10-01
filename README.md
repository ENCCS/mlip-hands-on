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

ENCCS is the Swedish node of the EuroCC 3 project, which has received funding
from the European High-Performance Computing Joint Undertaking (JU) under
Grant Agreement No. 101306701. The project is supported by the European
High-Performance Computing Joint Undertaking and its members. ENCCS also
receives national funding from Vinnova and the Swedish Research Council (VR).

Funded by the European Union. Views and opinions expressed are however those
of the author(s) only and do not necessarily reflect those of the European
Union or the granting authority (EuroHPC Joint Undertaking). Neither the
European Union nor the granting authority can be held responsible for them.
