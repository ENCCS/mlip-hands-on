"""UMA 1.2 small through fairchem: one predict unit, one ASE calculator per task."""

from __future__ import annotations

import time

import torch
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

CHECKPOINT = "uma-s-1p2"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class UMA:
    """Loads the checkpoint once; `calculator(task)` shares it between tasks."""

    def __init__(self, dtype: str = "float64"):
        settings = inference_settings_default()
        settings.base_precision_dtype = getattr(torch, dtype)
        start = time.perf_counter()
        self.unit = pretrained_mlip.get_predict_unit(CHECKPOINT, inference_settings=settings, device=device())
        self.load_s = time.perf_counter() - start
        self.dtype = dtype

    def calculator(self, task: str) -> FAIRChemCalculator:
        return FAIRChemCalculator(self.unit, task_name=task)


def versions() -> dict:
    from importlib.metadata import version

    out = {p: version(p) for p in ("torch", "fairchem-core", "ase", "numpy", "e3nn")}
    out["checkpoint"] = CHECKPOINT
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out
