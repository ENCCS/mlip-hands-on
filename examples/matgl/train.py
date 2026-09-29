"""Train M3GNet on EMT Cu data from scratch and by fine-tuning the MatPES model."""

from __future__ import annotations

import shutil

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule

from common import PES_MODELS, RESULTS, device, lightning_precision, load, save_json, timer
from dataset import cu_configs, loaders

N_STRUCTURES = 100


def scratch_module(mean_e: float) -> PotentialLightningModule:
    model = M3GNet(element_types=("Cu",), is_intensive=False, nblocks=2,
                   dim_node_embedding=32, dim_edge_embedding=32, units=32)
    return PotentialLightningModule(model=model, element_refs=np.array([mean_e]), lr=2e-3)


def finetune_module(structures, energies) -> PotentialLightningModule:
    """Pretrained M3GNet with the Cu reference shifted onto the EMT energy scale."""
    from matgl.ext.ase import PESCalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    pot = load(PES_MODELS["M3GNet"], device())
    calc = PESCalculator(pot)
    shift = []
    for s, e in zip(structures, energies):
        atoms = AseAtomsAdaptor().get_atoms(s)
        atoms.calc = calc
        shift.append((e - atoms.get_potential_energy()) / len(atoms))
    refs = pot.element_refs.property_offset.detach().cpu().numpy().copy()
    refs[pot.model.element_types.index("Cu")] += float(np.mean(shift))
    return PotentialLightningModule(model=pot.model, element_refs=refs, lr=1e-4,
                                    data_mean=pot.data_mean, data_std=pot.data_std)


def fit(module, loaders_, epochs: int, name: str) -> dict:
    train_l, val_l, test_l = loaders_
    logs = RESULTS / "train_logs"
    kw = {"accelerator": "auto", "devices": 1, "inference_mode": False, "enable_progress_bar": False,
          "enable_model_summary": False, "precision": lightning_precision()}
    before = L.Trainer(logger=False, **kw).test(module, dataloaders=test_l, verbose=False)[0]
    trainer = L.Trainer(max_epochs=epochs, logger=CSVLogger(logs, name=name, version=0), **kw)
    times: dict = {}
    with timer(times, "fit_s"):
        trainer.fit(module, train_dataloaders=train_l, val_dataloaders=val_l)
    after = trainer.test(module, dataloaders=test_l, verbose=False)[0]
    return {"epochs": epochs, "test_before": _mae(before), "test_after": _mae(after),
            "curve": _curve(logs / name / "version_0" / "metrics.csv"), **times}


def _mae(metrics: dict) -> dict:
    return {"energy_meV_atom": round(1000 * metrics["test_Energy_MAE"], 2),
            "force_eV_A": round(metrics["test_Force_MAE"], 4)}


def _curve(path) -> dict:
    df = pd.read_csv(path).groupby("epoch").first().dropna(subset=["train_Total_Loss"])
    cols = ["train_Total_Loss", "val_Total_Loss", "val_Energy_MAE", "val_Force_MAE"]
    return {c: df[c].round(6).tolist() for c in cols if c in df}


def main(epochs: int = 50) -> dict:
    torch.manual_seed(0)
    L.seed_everything(0, verbose=False)
    structures, labels = cu_configs(N_STRUCTURES)
    mean_e = float(np.mean([e / len(s) for s, e in zip(structures, labels["energies"])]))
    root = RESULTS / "graphs"
    shutil.rmtree(RESULTS / "train_logs", ignore_errors=True)
    out = {"data": "EMT-labelled fcc Cu, 32 atoms per cell", "n_structures": N_STRUCTURES,
           "split": [0.7, 0.15, 0.15], "runs": {}}
    scratch_l = loaders(structures, labels, ("Cu",), root)
    out["runs"]["scratch"] = fit(scratch_module(mean_e), scratch_l, epochs, "scratch")
    ft_types = load(PES_MODELS["M3GNet"], device()).model.element_types
    ft_l = loaders(structures, labels, ft_types, root)
    out["runs"]["finetune"] = fit(finetune_module(structures, labels["energies"]), ft_l, epochs, "finetune")
    shutil.rmtree(root, ignore_errors=True)
    for k, v in out["runs"].items():
        print(k, v["test_before"], "->", v["test_after"], f"{v['fit_s']} s")
    save_json(out, "train.json")
    return out
