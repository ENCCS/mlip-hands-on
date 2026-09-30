"""The three checks, for any backend: lattice constants (omat), charge and spin (omol), cost per force call."""

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


def outdir(model) -> Path:
    path = RESULTS / model.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(model, name: str, rows: list[dict]) -> None:
    with open(outdir(model) / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(model, name: str, data: dict) -> None:
    (outdir(model) / name).write_text(json.dumps(data, indent=2) + "\n")


def attach(atoms, calc) -> None:
    """ASE compares positions and cell but not atoms.info, so drop cached results before a new charge or spin."""
    calc.reset()
    atoms.calc = calc


def relax(atoms, calc, cell: bool = False):
    attach(atoms, calc)
    target = FrechetCellFilter(atoms) if cell else atoms
    BFGS(target, logfile=None).run(fmax=FMAX, steps=500)
    return atoms


def lattice(model) -> list[dict]:
    rows = []
    for name, spec in structures.CRYSTALS.items():
        atoms = relax(structures.crystal(name), model.calculator("omat"), cell=True)
        a = float(np.cbrt(atoms.get_volume()))
        rows.append(dict(crystal=name, model=model.checkpoint("omat"), a_A=round(a, 4), pbe_A=spec["pbe"], experiment_A=spec["experiment"],
                         vs_pbe_pct=round(100 * (a / spec["pbe"] - 1), 2),
                         vs_experiment_pct=round(100 * (a / spec["experiment"] - 1), 2)))
    save_csv(model, "lattice.csv", rows)
    return rows


def state(atoms) -> dict:
    return dict(energy_eV=atoms.get_potential_energy(), bond_A=round(atoms.get_distance(0, 1), 4))


def oxygen(model) -> dict:
    """Triplet ground state, closed-shell singlet, and the doublet cation at the triplet geometry."""
    calc = model.calculator("omol")
    triplet = relax(structures.oxygen(spin=3), calc)
    states = dict(triplet=state(triplet), singlet=state(relax(structures.oxygen(spin=1), calc)))
    cation = structures.oxygen(spin=2, charge=1)
    cation.positions = triplet.positions
    ie = single_point(cation, calc) - states["triplet"]["energy_eV"]
    gap = states["singlet"]["energy_eV"] - states["triplet"]["energy_eV"]
    return dict(states=states, gap_eV=round(gap, 3), experiment_eV=structures.O2_GAP_EV,
                ie_vertical_eV=round(ie, 3), ie_experiment_eV=structures.O2_IE_EV)


def single_point(atoms, calc) -> float:
    attach(atoms, calc)
    return atoms.get_potential_energy()


def geometry(atoms) -> dict:
    return dict(oh_A=round(atoms.get_distance(0, 1), 4), hoh_deg=round(atoms.get_angle(1, 0, 2), 1))


def water(model) -> dict:
    calc = model.calculator("omol")
    neutral = relax(structures.water(), calc)
    e_neutral = neutral.get_potential_energy()
    cation = structures.water(charge=1, spin=2)
    cation.positions = neutral.positions
    vertical = single_point(cation, calc) - e_neutral
    adiabatic = relax(cation, calc).get_potential_energy() - e_neutral
    return dict(vertical_eV=round(vertical, 3), adiabatic_eV=round(adiabatic, 3), experiment_eV=structures.WATER_IE_EV,
                neutral=geometry(neutral), cation=geometry(cation))


def molecules(model) -> dict:
    out = dict(model=model.checkpoint("omol"), o2=oxygen(model), water=water(model))
    save_json(model, "molecules.json", out)
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


def timing(model) -> list[dict]:
    systems = {
        "Si 216 atoms (omat)": (structures.crystal("Si").repeat(3), "omat"),
        "Cu 256 atoms (omat)": (structures.crystal("Cu").repeat(4), "omat"),
        "H2O 3 atoms (omol)": (structures.water(), "omol"),
    }
    rows = []
    for label, (atoms, task) in systems.items():
        row = dict(system=label, model=model.checkpoint(task), dtype=model.dtype, load_s="", first_call_s="",
                   per_call_s="")
        try:
            first, median = seconds_per_call(atoms, model.calculator(task))
            row.update(load_s=round(model.load_s[model.checkpoint(task)], 1), first_call_s=round(first, 3),
                       per_call_s=round(median, 4), status="ok")
        except RuntimeError as err:
            row["status"] = str(err).splitlines()[0][:80]
        rows.append(row)
    save_csv(model, f"timing_{model.dtype}.csv", rows)
    return rows
