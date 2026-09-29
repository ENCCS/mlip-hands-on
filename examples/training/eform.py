"""Train a small MEGNet or M3GNet formation-energy model on the MP 2018.6.1 data."""

import json
import time

import numpy as np
import torch
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, collate_fn_graph, split_dataset
from matgl.layers import BondExpansion
from matgl.models import M3GNet, MEGNet
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Structure

from . import config, report
from .fit import fit, loaders

CUTOFF = 4.0


def load(data, n: int):
    rows = json.loads((data / config.EFORM_SUBSET).read_text())[:n]
    return [Structure.from_str(r["structure"], fmt="cif") for r in rows], [r["eform"] for r in rows]


def build(name: str, elements):
    if name == "m3gnet":
        return M3GNet(element_types=elements, is_intensive=True, readout_type="set2set")
    return MEGNet(
        dim_node_embedding=16, dim_edge_embedding=100, dim_state_embedding=2, nblocks=3,
        hidden_layer_sizes_input=(64, 32), hidden_layer_sizes_conv=(64, 64, 32),
        nlayers_set2set=1, niters_set2set=2, hidden_layer_sizes_output=(32, 16),
        is_classification=False, activation_type="softplus2", element_types=elements,
        bond_expansion=BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5),
        cutoff=CUTOFF, gauss_width=0.5,
    )


def mae(module, loader, device: str) -> float:
    module = module.to(device).eval()
    errors = []
    with torch.no_grad():
        for g, lat, state_attr, y in loader:
            pred = module(g=g.to(device), lat=lat.to(device), state_attr=state_attr.to(device))
            pred = pred.reshape(-1).cpu() * module.data_std + module.data_mean
            errors.append((pred - y).abs())
    return round(torch.cat(errors).mean().item(), 4)


def run(args) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    structures, eform = load(args.data, args.n)
    elements = get_element_list(structures)
    dataset = MGLDataset(
        structures=structures, labels={"eform": eform}, save_cache=False, root=str(args.outdir / "graphs"),
        converter=Structure2Graph(element_types=elements, cutoff=CUTOFF),
    )
    splits = split_dataset(dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=args.seed)
    train_y = np.array([eform[i] for i in splits[0].indices])
    torch.manual_seed(args.seed)
    module = ModelLightningModule(
        model=build(args.model, elements), lr=args.lr, decay_steps=args.epochs,
        data_mean=float(train_y.mean()), data_std=float(train_y.std()),
    )
    train_l, val_l, test_l = loaders(*splits, collate_fn_graph, args.batch_size)
    start = time.perf_counter()
    curve = fit(module, train_l, val_l, epochs=args.epochs, outdir=args.outdir,
                name=f"eform-{args.model}", accelerator=args.accelerator, seed=args.seed)
    wall = time.perf_counter() - start
    device = "cuda" if torch.cuda.is_available() and args.accelerator != "cpu" else "cpu"
    record = {
        "model": args.model, "n_structures": len(structures), "n_elements": len(elements),
        "split": {"train": len(splits[0]), "val": len(splits[1]), "test": len(splits[2])},
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "train_wall_s": round(wall, 1),
        "mae_ev_atom": {k: mae(module, l, device) for k, l in (("train", train_l), ("val", val_l), ("test", test_l))},
        "curve": curve, "environment": report.environment(),
    }
    report.write_json(args.outdir / "eform.json", record)
    report.eform_figure(curve, args.outdir / "eform.png")
    print(json.dumps(record["mae_ev_atom"]), f"wall {wall:.0f} s")
