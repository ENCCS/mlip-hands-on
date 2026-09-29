"""Relax CsCl with each universal potential, then run a short NVT MD."""

from __future__ import annotations

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from matgl.ext.ase import MolecularDynamics, Relaxer
from pymatgen.io.ase import AseAtomsAdaptor

from common import PES_MODELS, device, load, save_json, timer
from prototypes import build


def relax(potential, structure, fmax: float = 0.01) -> dict:
    result = Relaxer(potential=potential).relax(structure, fmax=fmax)
    final = result["final_structure"]
    return {
        "a_final": round(final.lattice.a, 4),
        "energy_final": round(result["trajectory"].energies[-1], 4),
        "energies": [round(e, 4) for e in result["trajectory"].energies],
        "structure": final,
    }


def run_md(potential, structure, steps: int, temperature: int, seed: int = 0) -> dict:
    atoms = AseAtomsAdaptor().get_atoms(structure * (3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=_rng(seed))
    driver = MolecularDynamics(atoms, potential=potential, ensemble="nvt_langevin",
                               temperature=temperature, timestep=1.0, friction=0.01)
    log = {"energy_per_atom": [], "temperature": []}

    def record():
        log["energy_per_atom"].append(round(atoms.get_potential_energy() / len(atoms), 5))
        log["temperature"].append(round(atoms.get_temperature(), 2))

    driver.dyn.attach(record, interval=10)
    driver.run(steps)
    return {"n_atoms": len(atoms), "steps": steps, "interval": 10, **log}


def _rng(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def main(steps: int = 1000, temperature: int = 300) -> dict:
    start = build("cscl", ["Cs", "Cl"], 4.5)
    out = {"system": "CsCl", "a_start": 4.5, "temperature_K": temperature, "models": {}}
    for label, name in PES_MODELS.items():
        times: dict = {}
        pot = load(name, device())
        with timer(times, "relax_s"):
            rel = relax(pot, start)
        with timer(times, "md_s"):
            md = run_md(pot, rel.pop("structure"), steps, temperature)
        out["models"][label] = {"name": name, "relax": rel, "md": md, "timings": times}
        print(f"{label}: a = {rel['a_final']} A, E = {rel['energy_final']} eV, {times}")
    save_json(out, "relax_md.json")
    return out
