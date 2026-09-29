"""Small Cu dataset labelled with ASE's EMT potential (no database or API key)."""

from __future__ import annotations

from functools import partial

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes, split_dataset
from pymatgen.io.ase import AseAtomsAdaptor


def cu_configs(n: int, seed: int = 42) -> tuple[list, dict]:
    """Strained and rattled 32-atom fcc Cu cells with EMT energies and forces."""
    rng = np.random.default_rng(seed)
    structures, energies, forces = [], [], []
    for _ in range(n):
        atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
        atoms.set_cell(atoms.cell * rng.uniform(0.97, 1.03), scale_atoms=True)
        atoms.rattle(stdev=rng.uniform(0.02, 0.12), seed=int(rng.integers(1_000_000)))
        atoms.calc = EMT()
        energies.append(float(atoms.get_potential_energy()))
        forces.append(atoms.get_forces().tolist())
        structures.append(AseAtomsAdaptor().get_structure(atoms))
    return structures, {"energies": energies, "forces": forces}


def loaders(structures, labels, element_types, root, batch_size: int = 8):
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    data = MGLDataset(structures=structures, labels=labels, converter=converter,
                      root=str(root), save_cache=False, clear_processed=True)
    train, val, test = split_dataset(data, frac_list=[0.7, 0.15, 0.15], shuffle=True, random_state=42)
    collate = partial(collate_fn_pes, include_stress=False)
    return MGLDataLoader(train_data=train, val_data=val, test_data=test,
                         collate_fn=collate, batch_size=batch_size, num_workers=0)
