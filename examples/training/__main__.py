"""Command line: python -m training {prefetch,eform,finetune} [options]."""

import argparse
import os
from pathlib import Path

import matgl

from . import eform, finetune, prefetch


def training(sub, name: str, common, epochs: int, help: str) -> argparse.ArgumentParser:
    p = sub.add_parser(name, parents=[common], help=help)
    p.add_argument("--outdir", type=Path, default=Path("training_results"))
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--epochs", type=int, default=epochs)
    return p


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m training")
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=Path("training_data"))
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--element", default="Li", help="r2SCAN subset: structures containing this element")
    common.add_argument("--float-bits", type=int, choices=(32, 64), default=int(os.environ.get("MLIP_FLOAT_BITS", 32)),
                        help="floating-point width; 64 on LUMI, where some float32 kernels fail")

    pre = sub.add_parser("prefetch", parents=[common], help="download data and model (login node)")
    pre.add_argument("--n-eform", type=int, default=5000)
    pre.add_argument("--n-pes", type=int, default=1500)
    pre.add_argument("--max-sites", type=int, default=64)
    pre.add_argument("--skip-eform", action="store_true", help="skip the 1.2 GB formation-energy download")
    pre.set_defaults(func=prefetch.run)

    ef = training(sub, "eform", common, epochs=200, help="train a formation-energy model")
    ef.add_argument("--model", choices=("megnet", "m3gnet"), default="megnet")
    ef.add_argument("--n", type=int, default=5000)
    ef.add_argument("--lr", type=float, default=1e-3)
    ef.add_argument("--batch-size", type=int, default=64)
    ef.set_defaults(func=eform.run)

    ft = training(sub, "finetune", common, epochs=150, help="fine-tune versus from scratch")
    ft.add_argument("--fractions", type=float, nargs="+", default=list(finetune.FRACTIONS))
    ft.add_argument("--lr-finetune", type=float, default=2e-4)
    ft.add_argument("--lr-scratch", type=float, default=1e-3)
    ft.add_argument("--stress-weight", type=float, default=0.1)
    ft.add_argument("--batch-size", type=int, default=16)
    ft.set_defaults(func=finetune.run)
    return p


def main() -> None:
    args = parser().parse_args()
    matgl.set_default_dtype("float", args.float_bits)
    args.func(args)


if __name__ == "__main__":
    main()
