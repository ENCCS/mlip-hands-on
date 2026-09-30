"""Test systems and their reference values (sources on the lesson page)."""

from __future__ import annotations

from ase import Atoms
from ase.build import bulk, molecule

CRYSTALS = {
    "Si": dict(structure="diamond", a=5.43, experiment=5.431, pbe=5.470),
    "Cu": dict(structure="fcc", a=3.61, experiment=3.615, pbe=3.629),
}

O2_GAP_EV = 0.982
WATER_IE_EV = 12.62


def crystal(name: str) -> Atoms:
    spec = CRYSTALS[name]
    return bulk(name, spec["structure"], a=spec["a"], cubic=True)


def with_state(atoms: Atoms, charge: int, spin: int) -> Atoms:
    """UMA's omol task reads total charge and spin multiplicity from atoms.info."""
    atoms = atoms.copy()
    atoms.info.update(charge=charge, spin=spin)
    return atoms


def oxygen(spin: int) -> Atoms:
    return with_state(molecule("O2"), charge=0, spin=spin)


def water(charge: int = 0, spin: int = 1) -> Atoms:
    return with_state(molecule("H2O"), charge=charge, spin=spin)
