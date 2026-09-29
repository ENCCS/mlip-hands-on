"""Rattled copies of four small crystals: a toy screening set."""

import numpy as np
from ase import Atoms
from ase.build import bulk

FAMILIES = {
    "Cu-fcc": lambda: bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2)),
    "Si-diamond": lambda: bulk("Si", "diamond", a=5.43, cubic=True),
    "Fe-bcc": lambda: bulk("Fe", "bcc", a=2.87, cubic=True).repeat((2, 2, 2)),
    "Al-fcc": lambda: bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2)),
}


def build_workload(n_variants: int, seed: int = 42) -> tuple[list[Atoms], list[str]]:
    rng = np.random.default_rng(seed)
    structures, labels = [], []
    for name, make in FAMILIES.items():
        for i in range(n_variants):
            atoms = make()
            atoms.rattle(stdev=rng.uniform(0.03, 0.12), seed=int(rng.integers(1e6)))
            structures.append(atoms)
            labels.append(f"{name}-{i:02d}")
    return structures, labels
