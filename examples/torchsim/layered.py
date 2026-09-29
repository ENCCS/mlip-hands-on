"""Relax AB graphite with each model and compare the interlayer spacing.

Each model starts from two spacings (experiment and plain PBE) to expose flat
energy surfaces.

Run from the examples directory:  python -m torchsim.layered --device cpu
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from .models import MATGL_MODELS, ORB_MODELS, load_models
from .report import sha256

# Low-temperature experiment (Baskin and Meyer 1955) and plain PBE, both from
# Table I of Hazrati et al. 2014. PBE is the level of theory the models learn.
EXPERIMENT = {"a_A": 2.46, "d_A": 3.34}
PBE = {"a_A": 2.47, "d_A": 4.40}
VARIANTS = {
    "mace-small": ("mace-small", False),
    "orb-v3": ("orb-v3-conservative-inf-omat", False),
    "orb-v3+d3": ("orb-v3-conservative-inf-omat", True),
    "tensornet": ("matgl-tensornet-pbe", False),
}
DEFAULT_VARIANTS = ["mace-small", "orb-v3", "orb-v3+d3"]


@dataclass
class LayerResult:
    variant: str
    model: str
    d3: bool
    dtype: str
    start_d_A: float
    a_A: float
    c_A: float
    d_A: float
    d_error_pct: float
    steps: int
    converged: bool
    wall_s: float


def graphite(a: float = EXPERIMENT["a_A"], d: float = EXPERIMENT["d_A"]) -> Atoms:
    """Bernal (AB) graphite, P6_3/mmc, four atoms per cell."""
    frac = [(0, 0, 0.25), (0, 0, 0.75), (1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)]
    atoms = Atoms("C4", cell=[a, a, 2 * d, 90, 90, 120], pbc=True)
    atoms.set_scaled_positions(frac)
    return atoms


def relax_one(calc, start_d: float, fmax: float, max_steps: int) -> tuple:
    atoms = graphite(d=start_d)
    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    start = time.perf_counter()
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    a, _, c = atoms.cell.cellpar()[:3]
    return a, c, opt.nsteps, converged, time.perf_counter() - start


def relax(variant: str, device: torch.device, dtype: torch.dtype, fmax: float,
          max_steps: int, checkpoints: dict, starts: list[float]) -> list[LayerResult]:
    name, d3 = VARIANTS[variant]
    if name in MATGL_MODELS:
        dtype, checkpoint = torch.float32, None
    else:
        checkpoint = checkpoints["orb" if name in ORB_MODELS else "mace"]
    calc = load_models(name, device, dtype, checkpoint, d3).make_ase_calc()
    rows = []
    for d0 in starts:
        a, c, steps, converged, wall = relax_one(calc, d0, fmax, max_steps)
        error = 100 * (c / 2 - EXPERIMENT["d_A"]) / EXPERIMENT["d_A"]
        rows.append(LayerResult(variant, name, d3, str(dtype).removeprefix("torch."),
                                d0, a, c, c / 2, error, steps, converged, wall))
    return rows


def print_table(rows: list[LayerResult]) -> None:
    print(f"{'variant':12}{'start d':>8}{'a (A)':>8}{'d (A)':>8}{'d err':>8}"
          f"{'steps':>7}  converged")
    for r in rows:
        print(f"{r.variant:12}{r.start_d_A:>8.2f}{r.a_A:>8.3f}{r.d_A:>8.3f}"
              f"{r.d_error_pct:>7.1f}%{r.steps:>7}  {r.converged}")
    for label, ref in (("PBE", PBE), ("experiment", EXPERIMENT)):
        print(f"{label:12}{'':>8}{ref['a_A']:>8.3f}{ref['d_A']:>8.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS),
                   default=DEFAULT_VARIANTS)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                   help="MatGL always runs in float32")
    p.add_argument("--starts", nargs="+", type=float, default=[3.34, 4.4],
                   help="starting interlayer spacings, Angstrom")
    p.add_argument("--fmax", type=float, default=0.002, help="eV/Angstrom")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--mace-checkpoint", help="local MACE-MP-0b small file")
    p.add_argument("--orb-checkpoint", help="local Orb-v3 conservative-inf-omat file")
    p.add_argument("--outdir", default="graphite_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    auto = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(auto if args.device == "auto" else args.device)
    dtype = getattr(torch, args.dtype)
    checkpoints = {"mace": args.mace_checkpoint, "orb": args.orb_checkpoint}
    rows = [row for v in args.variants for row in relax(
        v, device, dtype, args.fmax, args.max_steps, checkpoints, args.starts)]
    print_table(rows)
    packages = {p: version(p) for p in ("torch", "ase", "mace-torch", "orb-models", "matgl")}
    summary = {"device": str(device), "dtype": args.dtype, "fmax": args.fmax,
               "starts_A": args.starts,
               "max_steps": args.max_steps, "experiment": EXPERIMENT, "pbe": PBE,
               "packages": packages,
               "checkpoint_sha256": {k: sha256(v) for k, v in checkpoints.items()},
               "results": [asdict(r) for r in rows]}
    (out / "graphite.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}/graphite.json")


if __name__ == "__main__":
    main()
