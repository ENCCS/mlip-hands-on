"""The three checks: lattice constants (omat), charge and spin (omol), and cost per force call."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

import structures

RESULTS = Path(__file__).resolve().parent / "results"
FMAX = 0.005


def save_csv(name: str, rows: list[dict]) -> None:
    with open(RESULTS / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(name: str, data: dict) -> None:
    (RESULTS / name).write_text(json.dumps(data, indent=2) + "\n")


def relax(atoms, calc, cell: bool = False):
    atoms.calc = calc
    target = FrechetCellFilter(atoms) if cell else atoms
    BFGS(target, logfile=None).run(fmax=FMAX, steps=500)
    return atoms


def lattice(uma) -> list[dict]:
    rows = []
    for name, spec in structures.CRYSTALS.items():
        atoms = relax(structures.crystal(name), uma.calculator("omat"), cell=True)
        a = float(np.cbrt(atoms.get_volume()))
        rows.append(dict(crystal=name, uma_A=round(a, 4), pbe_A=spec["pbe"], experiment_A=spec["experiment"],
                         vs_pbe_pct=round(100 * (a / spec["pbe"] - 1), 2),
                         vs_experiment_pct=round(100 * (a / spec["experiment"] - 1), 2)))
    save_csv("lattice.csv", rows)
    return rows


def oxygen(uma) -> dict:
    calc = uma.calculator("omol")
    states = {}
    for label, spin in (("triplet", 3), ("singlet", 1)):
        atoms = relax(structures.oxygen(spin), calc)
        states[label] = dict(energy_eV=atoms.get_potential_energy(), bond_A=round(atoms.get_distance(0, 1), 4))
    gap = states["singlet"]["energy_eV"] - states["triplet"]["energy_eV"]
    return dict(states=states, gap_eV=round(gap, 3), experiment_eV=structures.O2_GAP_EV)


def single_point(atoms, calc) -> float:
    atoms.calc = calc
    return atoms.get_potential_energy()


def water(uma) -> dict:
    calc = uma.calculator("omol")
    neutral = relax(structures.water(), calc)
    cation = structures.water(charge=1, spin=2)
    cation.positions = neutral.positions
    vertical = single_point(cation, calc) - neutral.get_potential_energy()
    adiabatic = relax(cation, calc).get_potential_energy() - neutral.get_potential_energy()
    return dict(vertical_eV=round(vertical, 3), adiabatic_eV=round(adiabatic, 3),
                experiment_eV=structures.WATER_IE_EV)


def molecules(uma) -> dict:
    out = dict(o2=oxygen(uma), water=water(uma))
    save_json("molecules.json", out)
    return out


def seconds_per_call(atoms, calc, calls: int = 20) -> tuple[float, float]:
    """First call (includes warm-up), then the median of later calls on rattled copies."""
    times = []
    for i in range(calls + 1):
        trial = atoms.copy()
        trial.rattle(0.01, seed=i)
        trial.calc = calc
        start = time.perf_counter()
        trial.get_forces()
        times.append(time.perf_counter() - start)
    return times[0], float(np.median(times[1:]))


def timing(uma) -> list[dict]:
    systems = {
        "Si 216 atoms (omat)": (structures.crystal("Si").repeat(3), "omat"),
        "Cu 256 atoms (omat)": (structures.crystal("Cu").repeat(4), "omat"),
        "H2O 3 atoms (omol)": (structures.water(), "omol"),
    }
    rows = []
    for label, (atoms, task) in systems.items():
        row = dict(system=label, dtype=uma.dtype, load_s=round(uma.load_s, 1), first_call_s="", per_call_s="")
        try:
            first, median = seconds_per_call(atoms, uma.calculator(task))
            row.update(first_call_s=round(first, 3), per_call_s=round(median, 4), status="ok")
        except RuntimeError as err:
            row["status"] = str(err).splitlines()[0][:80]
        rows.append(row)
    return rows
