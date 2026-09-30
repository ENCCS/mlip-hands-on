"""Three universal MLIPs as ASE calculators, all in float64."""

from __future__ import annotations

import torch

torch.set_default_dtype(torch.float64)

MODELS = {
    "MACE-MP-0b": "mace-mp-0b small (MPtrj, PBE and PBE+U)",
    "Orb-v3": "orb-v3-conservative-inf-omat (OMat24, PBE and PBE+U)",
    "TensorNet": "TensorNet-PES-MatPES-PBE-2025.2 (MatPES, PBE)",
}
MACE_MODEL = "small-0b"
MATGL_MODEL = "TensorNet-PES-MatPES-PBE-2025.2"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def mace():
    from mace.calculators import mace_mp

    return mace_mp(model=MACE_MODEL, device=device(), default_dtype="float64")


def orb():
    """orb-models 0.5.x API: 0.6 and later need Python 3.12, the LUMI module has 3.11."""
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    model = pretrained.orb_v3_conservative_inf_omat(device=device(), precision="float64", compile=False)
    return ORBCalculator(model, device=device())


def tensornet():
    import matgl
    from matgl.ext.ase import PESCalculator

    matgl.set_default_dtype("float", 64)
    potential = matgl.load_model(MATGL_MODEL).to(device=device(), dtype=torch.float64)
    return PESCalculator(potential, stress_unit="eV/A3")


LOADERS = {"MACE-MP-0b": mace, "Orb-v3": orb, "TensorNet": tensornet}


def load(name: str):
    """Return an ASE calculator that counts its own calculate() calls in `.calls`."""
    calc = LOADERS[name]()
    calc.calls = 0
    calculate = calc.calculate

    def counted(*args, **kwargs):
        calc.calls += 1
        return calculate(*args, **kwargs)

    calc.calculate = counted
    return calc


def versions() -> dict:
    from importlib.metadata import version

    pkgs = ("torch", "ase", "pymatgen", "mace-torch", "orb-models", "matgl")
    out = {p: version(p) for p in pkgs}
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    out["float"] = str(torch.get_default_dtype()).removeprefix("torch.")
    return out
