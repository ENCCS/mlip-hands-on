"""Shared helpers: paths, model loading, timing and JSON output."""

from __future__ import annotations

import json
import os
import time
import warnings
from contextlib import contextmanager
from functools import cache
from pathlib import Path

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parents[1] / "_static"

PES_MODELS = {
    "TensorNet": "TensorNet-PES-MatPES-PBE-2025.2",
    "M3GNet": "M3GNet-PES-MatPES-PBE-2025.2",
}
# The MatGL tutorial uses the 2018.6.1 weights; the newer 2019.4.1 release is used here.
EFORM_MODEL = "M3GNet-Eform-MP-2019.4.1"
GAP_MODEL = "MEGNet-BandGap-mfi-MP-2019.4.1"
# 32 or 64. Use 64 on LUMI: float32 det/prod kernels fail there (ROCm "CUDA driver error: 209").
FLOAT_BITS = int(os.environ.get("MATGL_FLOAT_BITS", "32"))


def set_precision(bits: int = FLOAT_BITS) -> None:
    """Set the float size for matgl graphs, labels and new models; call before loading models."""
    import matgl

    matgl.set_default_dtype("float", bits)


def lightning_precision() -> str:
    import torch

    return "64-true" if torch.get_default_dtype() == torch.float64 else "32-true"


@cache
def load(name: str, device: str = "cpu"):
    """Load a pretrained model in the default float dtype; potentials go to the GPU, property models stay on the CPU."""
    import matgl
    import torch

    return matgl.load_model(name).to(device=device, dtype=torch.get_default_dtype())


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def timer(store: dict, key: str):
    start = time.perf_counter()
    yield
    store[key] = round(time.perf_counter() - start, 2)


def versions() -> dict:
    from importlib.metadata import version

    import torch

    pkgs = ("matgl", "torch", "lightning", "torch_geometric", "ase", "pymatgen")
    out = {p: version(p) for p in pkgs}
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    out["float"] = str(torch.get_default_dtype()).removeprefix("torch.")
    return out


def save_json(data: dict, name: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(data, indent=2))
    return path


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())
