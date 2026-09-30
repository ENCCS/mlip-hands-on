"""Relax both end points at fixed cell, then run CI-NEB for the Li hop; record barrier, force calls and time."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import write
from ase.mep import NEB, NEBTools
from ase.optimize import BFGS, FIRE

import models
import structures

RESULTS = Path(__file__).resolve().parent / "results"
OPTIMISERS = {"FIRE": FIRE, "BFGS": BFGS}


class ImageCalculator(Calculator):
    """Per-image view of one shared model, so each image caches its own results.

    With one calculator shared by all images, ASE re-evaluates every image each time
    the optimiser asks the band for forces (about three times per step).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        copy = self.atoms.copy()
        copy.calc = self.model
        self.results = {"energy": copy.get_potential_energy(), "forces": copy.get_forces()}


def freeze(atoms) -> None:
    """Store the current energy and forces so later reads do not call the model again."""
    atoms.calc = SinglePointCalculator(atoms, energy=atoms.get_potential_energy(),
                                       forces=atoms.get_forces())


def lateral_offset(band, hopper: int) -> list[list[float]]:
    """Offset (x, z) in Å of the hopping Li from the straight line joining its two sites."""
    start = band[0].positions[hopper]
    line, _ = find_mic(band[-1].positions[hopper] - start, band[0].cell)
    unit = line / np.linalg.norm(line)
    out = []
    for atoms in band:
        d, _ = find_mic(atoms.positions[hopper] - start, atoms.cell)
        perp = d - (d @ unit) * unit
        out.append([round(float(perp[0]), 4), round(float(perp[2]), 4)])
    return out


def run(name: str, images: int = 7, fmax: float = 0.05, optimiser: str = "FIRE",
        steps: int = 1000) -> dict:
    opt = OPTIMISERS[optimiser]
    calc = models.load(name)
    initial, final, hopper = structures.end_points()

    start = time.perf_counter()
    relaxed = []
    for atoms in (initial, final):
        atoms.calc = calc
        relaxed.append(bool(opt(atoms, logfile=None).run(fmax=fmax, steps=steps)))
        freeze(atoms)
    relax = {"converged": relaxed, "force_calls": calc.calls,
             "wall_s": round(time.perf_counter() - start, 2)}

    band = [initial] + [initial.copy() for _ in range(images)] + [final]
    for atoms in band[1:-1]:
        atoms.calc = ImageCalculator(calc)
    neb = NEB(band, climb=True)
    neb.interpolate(method="idpp", mic=True)
    calc.calls = 0
    start = time.perf_counter()
    optimizer = opt(neb, logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    neb_run = {"converged": converged, "steps": optimizer.nsteps, "force_calls": calc.calls,
               "wall_s": round(time.perf_counter() - start, 2)}
    for atoms in band[1:-1]:
        freeze(atoms)

    tools = NEBTools(band)
    barrier, delta = tools.get_barrier(fit=False)
    fit = tools.get_fit()
    e0 = band[0].get_potential_energy()
    result = {
        "model": name, "description": models.MODELS[name], "atoms": len(initial),
        "images": images, "fmax": fmax, "optimiser": optimiser,
        "barrier_eV": round(float(barrier), 4), "reaction_energy_eV": round(float(delta), 4),
        "hop_distance_A": round(float(fit.path[-1]), 4),
        "relax": relax, "neb": neb_run,
        "path_A": [round(float(x), 4) for x in fit.path],
        "energies_eV": [round(float(a.get_potential_energy() - e0), 5) for a in band],
        "fit_path_A": [round(float(x), 4) for x in fit.fit_path],
        "fit_energies_eV": [round(float(x), 5) for x in fit.fit_energies],
        "li_offset_xz_A": lateral_offset(band, hopper),
        "hopper": hopper,
    }
    RESULTS.mkdir(exist_ok=True)
    stem = name.lower().replace("-", "_")
    (RESULTS / f"{stem}.json").write_text(json.dumps(result, indent=2))
    write(RESULTS / f"{stem}_band.extxyz", band)
    return result
