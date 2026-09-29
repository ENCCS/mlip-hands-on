"""Load one potential for both the ASE baseline and the TorchSim batch."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

MACE_MODELS = {"mace-small": "mace_mp_small", "mace-mpa": "mace_mpa_medium"}


@dataclass
class ModelPair:
    torchsim: object
    make_ase_calc: Callable
    checkpoint: str | None = None


def load_lj(device: torch.device, dtype: torch.dtype) -> ModelPair:
    """Toy Lennard-Jones potential for offline smoke tests only."""
    from ase.calculators.lj import LennardJones
    from torch_sim.models.lennard_jones import LennardJonesModel

    model = LennardJonesModel(sigma=2.0, epsilon=0.1, device=device, dtype=dtype)
    return ModelPair(model, lambda: LennardJones(sigma=2.0, epsilon=0.1))


def load_mace(name: str, device: torch.device, dtype: torch.dtype,
              checkpoint: str | None = None) -> ModelPair:
    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel, MaceUrls

    source = checkpoint or getattr(MaceUrls, MACE_MODELS[name]).value
    options = {"model": source, "device": str(device),
               "default_dtype": str(dtype).removeprefix("torch.")}
    raw = mace_mp(return_raw_model=True, **options)
    model = MaceModel(model=raw, device=device, dtype=dtype,
                      compute_forces=True, compute_stress=False)
    return ModelPair(model, lambda: mace_mp(**options), checkpoint)


def load_models(name: str, device: torch.device, dtype: torch.dtype,
                checkpoint: str | None = None) -> ModelPair:
    if name == "lj":
        return load_lj(device, dtype)
    return load_mace(name, device, dtype, checkpoint)
