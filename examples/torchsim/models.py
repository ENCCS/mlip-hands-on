"""Load one potential for both the ASE baseline and the TorchSim batch."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

MACE_MODELS = {"mace-small": "mace_mp_small", "mace-mpa": "mace_mpa_medium"}
ORB_MODELS = {"orb-v3-conservative-inf-omat": "orb_v3_conservative_inf_omat"}
ORB_PRECISION = {torch.float64: "float64", torch.float32: "float32-highest"}


@dataclass
class ModelPair:
    make_torchsim: Callable
    make_ase_calc: Callable
    checkpoint: str | None = None


@contextmanager
def default_dtype(dtype: torch.dtype | None = None) -> Iterator[None]:
    """Set torch's global default dtype for a block, then restore it."""
    saved = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved)


def keeps_dtype(factory: Callable) -> Callable:
    """MACE and Orb loaders change the global default dtype; undo that."""
    def build():
        with default_dtype():
            return factory()
    return build


def load_lj(device: torch.device, dtype: torch.dtype) -> ModelPair:
    """Toy Lennard-Jones potential for offline smoke tests only."""
    from ase.calculators.lj import LennardJones
    from torch_sim.models.lennard_jones import LennardJonesModel

    return ModelPair(
        lambda: LennardJonesModel(sigma=2.0, epsilon=0.1, device=device, dtype=dtype),
        lambda: LennardJones(sigma=2.0, epsilon=0.1))


def load_mace(name: str, device: torch.device, dtype: torch.dtype,
              checkpoint: str | None = None) -> ModelPair:
    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel, MaceUrls

    source = checkpoint or getattr(MaceUrls, MACE_MODELS[name]).value
    options = {"model": source, "device": str(device),
               "default_dtype": str(dtype).removeprefix("torch.")}

    def make_torchsim():
        raw = mace_mp(return_raw_model=True, **options)
        return MaceModel(model=raw, device=device, dtype=dtype,
                         compute_forces=True, compute_stress=False)
    return ModelPair(keeps_dtype(make_torchsim),
                     keeps_dtype(lambda: mace_mp(**options)), checkpoint)


def orb_ase_calc(orbff, adapter, device: torch.device, dtype: torch.dtype):
    """ORBCalculator builds its input graph in the global default dtype."""
    from orb_models.forcefield.inference.calculator import ORBCalculator

    class Calculator(ORBCalculator):
        def calculate(self, *args, **kwargs):
            with default_dtype(dtype):
                super().calculate(*args, **kwargs)
    return Calculator(orbff, adapter, device=device)


def load_orb(name: str, device: torch.device, dtype: torch.dtype,
             checkpoint: str | None = None, d3: bool = False) -> ModelPair:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.d3_model import AlchemiDFTD3, D3SumModel
    from torch_sim.models.orb import OrbModel

    loader = getattr(pretrained, ORB_MODELS[name])
    options = {"device": device, "precision": ORB_PRECISION[dtype], "compile": False}
    if checkpoint:
        options["weights_path"] = checkpoint
    with default_dtype():
        orbff, adapter = loader(**options)
    if d3:
        orbff = D3SumModel(orbff, AlchemiDFTD3(functional="PBE", damping="BJ"))
    return ModelPair(lambda: OrbModel(orbff, adapter, device=device, dtype=dtype),
                     lambda: orb_ase_calc(orbff, adapter, device, dtype), checkpoint)


def load_models(name: str, device: torch.device, dtype: torch.dtype,
                checkpoint: str | None = None, d3: bool = False) -> ModelPair:
    if d3 and name not in ORB_MODELS:
        raise ValueError("D3 is wired up for Orb models only")
    if name == "lj":
        return load_lj(device, dtype)
    if name in ORB_MODELS:
        return load_orb(name, device, dtype, checkpoint, d3)
    return load_mace(name, device, dtype, checkpoint)
