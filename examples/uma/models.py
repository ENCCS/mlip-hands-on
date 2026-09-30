"""Two backends with one interface: `calculator(task)` for task "omat" (crystals) or "omol" (molecules)."""

from __future__ import annotations

import time

import torch


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Backend:
    """Loads checkpoints on first use and records the load time of each."""

    checkpoints: dict[str, str] = {}

    def __init__(self, dtype: str = "float64"):
        self.dtype = dtype
        self.load_s: dict[str, float] = {}
        self._calcs = {}

    def checkpoint(self, task: str) -> str:
        return self.checkpoints[task]

    def calculator(self, task: str):
        if task not in self._calcs:
            start = time.perf_counter()
            self._calcs[task] = self._load(task)
            self.load_s.setdefault(self.checkpoint(task), time.perf_counter() - start)
        return self._calcs[task]

    def _load(self, task: str):
        raise NotImplementedError


class Orb(Backend):
    """Orb-v3 conservative (OMat24) for crystals, OrbMol (OMol25, charge and spin) for molecules."""

    name = "orb"
    checkpoints = {"omat": "orb-v3-conservative-inf-omat", "omol": "orb-v3-conservative-omol"}

    def _load(self, task: str):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        precision = "float64" if self.dtype == "float64" else "float32-highest"
        loader = getattr(pretrained, self.checkpoint(task).replace("-", "_"))
        model = loader(device=device(), precision=precision, compile=False)
        return ORBCalculator(model, device=device())


class UMA(Backend):
    """UMA 1.2 small: one predict unit shared by all tasks."""

    name = "uma"
    checkpoints = {"omat": "uma-s-1p2", "omol": "uma-s-1p2"}

    def _load(self, task: str):
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

        if not hasattr(self, "unit"):
            settings = inference_settings_default()
            settings.base_precision_dtype = getattr(torch, self.dtype)
            self.unit = pretrained_mlip.get_predict_unit(self.checkpoint(task), inference_settings=settings,
                                                         device=device())
        return FAIRChemCalculator(self.unit, task_name=task)


BACKENDS = {"orb": Orb, "uma": UMA}
PACKAGES = {"orb": ("orb-models",), "uma": ("fairchem-core", "e3nn")}


def versions(name: str) -> dict:
    from importlib.metadata import version

    out = {p: version(p) for p in ("torch", "ase", "numpy", *PACKAGES[name])}
    out["checkpoints"] = sorted(set(BACKENDS[name].checkpoints.values()))
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out
