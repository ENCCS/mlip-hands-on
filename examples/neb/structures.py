"""Olivine LiFePO4 supercell with one Li vacancy and the two end points of a Li hop along [010]."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

# Pnma cell and sites of LiFePO4 from X-ray diffraction (Streltsov et al., 1993; COD 2100916).
CELL = (10.332, 6.010, 4.692)
SITES = [
    ("Li", (0.0, 0.0, 0.0)),
    ("Fe", (0.28222, 0.25, 0.97472)),
    ("P", (0.09486, 0.25, 0.41820)),
    ("O", (0.09678, 0.25, 0.7428)),
    ("O", (0.45710, 0.25, 0.2060)),
    ("O", (0.16558, 0.04646, 0.2848)),
]
SUPERCELL = (1, 2, 2)


def unit_cell() -> Atoms:
    species, coords = zip(*SITES)
    s = Structure.from_spacegroup("Pnma", Lattice.orthorhombic(*CELL), species, coords)
    return AseAtomsAdaptor.get_atoms(s)


def end_points(supercell: tuple[int, int, int] = SUPERCELL) -> tuple[Atoms, Atoms, int]:
    """Remove the Li at the origin; the neighbouring Li along b hops into that vacancy.

    Returns the initial and final images (same atom order) and the index of the hopping Li.
    """
    atoms = unit_cell().repeat(supercell)
    atoms.wrap()
    li = [a.index for a in atoms if a.symbol == "Li"]
    vacancy = min(li, key=lambda i: np.linalg.norm(atoms.positions[i]))
    target = atoms.positions[vacancy] + [0, CELL[1] / 2, 0]
    hopper = min(li, key=lambda i: np.linalg.norm(atoms.positions[i] - target))
    site = atoms.positions[vacancy].copy()
    del atoms[vacancy]
    hopper -= hopper > vacancy
    final = atoms.copy()
    final.positions[hopper] = site
    return atoms, final, hopper
