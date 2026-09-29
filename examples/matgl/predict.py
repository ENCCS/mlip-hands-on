"""Relax with a universal potential, then predict formation energy and band gap."""

from __future__ import annotations

import torch
from matgl.ext.ase import Relaxer

from common import EFORM_MODEL, GAP_MODEL, PES_MODELS, device, load, save_json, timer
from prototypes import build

A_START = 4.5
FIDELITIES = {"PBE": 0, "GLLB-SC": 1, "HSE": 2, "SCAN": 3}
# Approximate experimental references: formation enthalpy at 298 K (eV/atom, from
# CRC Handbook kJ/mol values) and room-temperature band gap (eV).
CRYSTALS = {
    "SrTiO3": ("perovskite", ["Sr", "Ti", "O"], -3.47, 3.25),
    "MgO": ("rocksalt", ["Mg", "O"], -3.12, 7.8),
    "NaCl": ("rocksalt", ["Na", "Cl"], -2.13, 8.5),
    "GaAs": ("zincblende", ["Ga", "As"], -0.37, 1.42),
    "Si": ("diamond", ["Si"], 0.0, 1.12),
}


def properties(structure) -> dict:
    eform = float(load(EFORM_MODEL).predict_structure(structure))
    gap = load(GAP_MODEL)
    gaps = {k: float(gap.predict_structure(structure, state_attr=torch.tensor([v])))
            for k, v in FIDELITIES.items()}
    return {"eform_eV_atom": round(eform, 3), **{f"gap_{k}_eV": round(g, 2) for k, g in gaps.items()}}


def one_crystal(relaxer: Relaxer, prototype: str, species: list[str]) -> dict:
    start = build(prototype, species, A_START)
    final = relaxer.relax(start, fmax=0.01, steps=1000)["final_structure"]
    return {"a_relaxed_A": round(final.lattice.abc[0], 4),
            "unrelaxed": properties(start), "relaxed": properties(final)}


def main(potential: str = "TensorNet") -> dict:
    relaxer = Relaxer(potential=load(PES_MODELS[potential], device()))
    out = {"relaxer": PES_MODELS[potential], "eform_model": EFORM_MODEL,
           "gap_model": GAP_MODEL, "a_start_A": A_START, "timings": {}, "crystals": {}}
    with timer(out["timings"], "total_s"):
        for name, (proto, species, h_expt, gap_expt) in CRYSTALS.items():
            res = one_crystal(relaxer, proto, species)
            out["crystals"][name] = {"eform_expt_eV_atom": h_expt, "gap_expt_eV": gap_expt, **res}
            print(name, res)
    save_json(out, "predict.json")
    return out
