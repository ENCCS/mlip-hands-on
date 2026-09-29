"""Relax a screening set serially with ASE, then in one TorchSim batch.

Run from the examples directory:  python -m torchsim --model lj --n-variants 2
"""

import argparse
from pathlib import Path

import torch

from .models import MACE_MODELS, ORB_MODELS, load_models
from .report import print_summary, summarise, write_outputs
from .runners import BatchedRelaxer, SerialRelaxer
from .workload import build_workload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["lj", *MACE_MODELS, *ORB_MODELS], default="mace-small")
    p.add_argument("--checkpoint",
                   help="local weights file; replaces the download for --model")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--n-variants", type=int, default=16, help="per crystal family")
    p.add_argument("--baseline-n", type=int, default=8)
    p.add_argument("--fmax", type=float, default=0.05, help="eV/Angstrom")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--autobatch", action="store_true")
    p.add_argument("--outdir", default="torchsim_results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.model == "lj" and args.checkpoint:
        p.error("--checkpoint needs a MACE or Orb --model")
    return args


def main() -> None:
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    auto = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(auto if args.device == "auto" else args.device)
    dtype = getattr(torch, args.dtype)
    structures, labels = build_workload(args.n_variants, args.seed)
    n_base = min(args.baseline_n, len(structures))
    print(f"model={args.model} device={device} dtype={args.dtype} "
          f"structures={len(structures)} serial subset={n_base}")

    models = load_models(args.model, device, dtype, args.checkpoint)
    serial = SerialRelaxer(models.make_ase_calc, args.fmax, args.max_steps)
    batched = BatchedRelaxer(models.torchsim, args.fmax, args.max_steps, args.autobatch)
    serial_result = serial.run(structures[:n_base])
    batched_result = batched.run(structures)

    config = {"model": args.model, "device": str(device), "dtype": args.dtype,
              "checkpoint": Path(args.checkpoint).name if args.checkpoint else None,
              "n_atoms_total": sum(len(a) for a in structures),
              "autobatch": args.autobatch, "fmax": args.fmax,
              "max_steps": args.max_steps}
    summary = summarise(config, labels, serial_result, batched_result,
                        models.checkpoint)
    print_summary(summary)
    write_outputs(out, summary, batched_result.relaxed)
    print(f"wrote {out}/summary.json and {out}/relaxed.extxyz")


if __name__ == "__main__":
    main()
