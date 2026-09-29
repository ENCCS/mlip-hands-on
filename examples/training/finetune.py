"""Fine-tune the PBE TensorNet on r2SCAN data and compare with training from scratch."""

import json
import time
from functools import partial

import torch
from matgl.graph.data import collate_fn_pes, split_dataset
from matgl.utils.training import MGLDatasetLoader, PotentialLightningModule
from torch.utils.data import Subset

from . import config, potential, report
from .fit import fit, loaders

FRACTIONS = (0.1, 0.5, 1.0)


def dataset(args, element_types):
    return MGLDatasetLoader.from_json(
        config.pes_subset(args.data, args.element), element_types=element_types,
        save_cache=False, root=str(args.outdir / "graphs"),
    )


def train_case(args, kind, model, base, refs, train, val, collate):
    lr = args.lr_finetune if kind == "fine-tuned" else args.lr_scratch
    module = PotentialLightningModule(
        model=model, element_refs=refs, data_mean=base.data_mean, data_std=base.data_std,
        stress_weight=args.stress_weight, loss="huber_loss", lr=lr, decay_steps=args.epochs,
    )
    train_l, val_l = loaders(train, val, None, collate, args.batch_size)
    start = time.perf_counter()
    curve = fit(module, train_l, val_l, epochs=args.epochs, outdir=args.outdir,
                name=f"{kind}-{len(train)}", accelerator=args.accelerator, seed=args.seed)
    return module.model, {"lr": lr, "train_wall_s": round(time.perf_counter() - start, 1), "curve": curve}


def run(args) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and args.accelerator != "cpu" else "cpu"
    base = potential.pretrained(args.data)
    refs = potential.atom_refs(args.data, base.model.element_types)
    train, val, test = split_dataset(dataset(args, base.model.element_types),
                                     frac_list=[0.7, 0.1, 0.2], shuffle=True, random_state=args.seed)
    collate = partial(collate_fn_pes, include_stress=True)
    test_l = loaders(test, test, None, collate, args.batch_size)[1]
    cases = [
        {"name": "zero-shot (PBE)", "kind": "zero-shot", "n_train": 0, **potential.evaluate(base, test_l, device)},
        {"name": "zero-shot, r2SCAN atom energies", "kind": "zero-shot", "n_train": 0,
         **potential.evaluate(potential.with_refs(base, refs), test_l, device)},
    ]
    for fraction in args.fractions:
        subset = Subset(train, range(max(2, round(fraction * len(train)))))
        for kind in ("fine-tuned", "scratch"):
            torch.manual_seed(args.seed)
            model = potential.pretrained(args.data).model if kind == "fine-tuned" else potential.fresh_model(base)
            trained, info = train_case(args, kind, model, base, refs, subset, val, collate)
            cases.append({"name": f"{kind}, {fraction:.0%}", "kind": kind, "fraction": fraction,
                          "n_train": len(subset), **potential.evaluate(trained, test_l, device), **info})
            print(json.dumps({k: v for k, v in cases[-1].items() if k != "curve"}))
    record = {
        "pretrained": config.PRETRAINED, "pool": config.MATPES_POOL, "element": args.element,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "epochs": args.epochs, "batch_size": args.batch_size, "stress_weight": args.stress_weight,
        "seed": args.seed, "cases": cases, "environment": report.environment(),
    }
    report.write_json(args.outdir / "finetune.json", record)
    report.finetune_figure(cases, args.outdir / "finetune.png")
