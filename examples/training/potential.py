"""Pretrained, re-referenced and freshly initialised TensorNet potentials, and their test errors."""

import json
from pathlib import Path

import matgl
import numpy as np
import torch
from matgl.apps.pes import Potential

from . import config


def pretrained(data: Path) -> Potential:
    return matgl.load_model(str(config.model_dir(data)))


def atom_refs(data: Path, element_types) -> np.ndarray:
    lines = config.atom_refs(data).read_text().splitlines()
    energies = {r["chemsys"]: r["energy"] for r in map(json.loads, lines)}
    return np.array([energies.get(el, 0.0) for el in element_types])


def with_refs(base: Potential, refs: np.ndarray) -> Potential:
    """Same network, different isolated-atom energies."""
    return Potential(model=base.model, element_refs=refs, data_mean=base.data_mean, data_std=base.data_std)


def fresh_model(base: Potential) -> torch.nn.Module:
    """Same architecture and hyperparameters, random weights."""
    return type(base.model)(**base.model._init_args)


def evaluate(potential: Potential, loader, device: str) -> dict[str, float]:
    potential = potential.to(device).eval()
    e_err, f_err, s_err, n_struct, n_force, n_stress = 0.0, 0.0, 0.0, 0, 0, 0
    for g, lat, state_attr, energy, forces, stress in loader:
        g, lat, state_attr = g.to(device), lat.to(device), state_attr.to(device)
        with torch.enable_grad():
            e_pred, f_pred, s_pred, _ = potential(g=g, lat=lat, state_attr=state_attr)
        n_atoms = torch.bincount(g.batch).cpu()
        e_err += ((e_pred.detach().cpu() - energy).abs() / n_atoms).sum().item()
        f_err += (f_pred.detach().cpu() - forces).abs().sum().item()
        s_err += (s_pred.detach().cpu().reshape(stress.shape) - stress).abs().sum().item()
        n_struct, n_force, n_stress = n_struct + len(energy), n_force + forces.numel(), n_stress + stress.numel()
    return {
        "energy_mae_mev_atom": round(1000 * e_err / n_struct, 2),
        "force_mae_mev_A": round(1000 * f_err / n_force, 2),
        "stress_mae_gpa": round(s_err / n_stress, 4),
    }
