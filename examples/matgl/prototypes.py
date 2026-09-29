"""Cubic prototype crystals built from space groups, no database needed."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

PROTOTYPES = {
    "fcc": ("Fm-3m", [[0, 0, 0]]),
    "bcc": ("Im-3m", [[0, 0, 0]]),
    "diamond": ("Fd-3m", [[0, 0, 0]]),
    "zincblende": ("F-43m", [[0, 0, 0], [0.25, 0.25, 0.25]]),
    "rocksalt": ("Fm-3m", [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "cscl": ("Pm-3m", [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "perovskite": ("Pm-3m", [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]),
}


def build(prototype: str, species: list[str], a: float) -> Structure:
    """Conventional cubic cell; species are listed in Wyckoff-site order."""
    group, coords = PROTOTYPES[prototype]
    return Structure.from_spacegroup(group, Lattice.cubic(a), species, coords)
