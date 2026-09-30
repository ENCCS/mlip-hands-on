"""Command-line entry: python cli.py {prefetch,versions,run,timing} --model {orb,uma}."""

from __future__ import annotations

import argparse
import json
import warnings

warnings.simplefilter("ignore")

import models
import tasks

STEPS = ["prefetch", "versions", "run", "timing"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--model", choices=list(models.BACKENDS), default="orb")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()
    model = models.BACKENDS[args.model](args.dtype)

    if args.step == "prefetch":
        for task in ("omat", "omol"):
            model.calculator(task)
        print("cached", model.load_s)
    elif args.step == "versions":
        text = json.dumps(models.versions(args.model), indent=2)
        (tasks.outdir(model) / "versions.json").write_text(text + "\n")
        print(text)
    elif args.step == "run":
        for row in tasks.lattice(model):
            print(f"{row['crystal']}: a = {row['a_A']} Å (PBE {row['pbe_A']}, experiment {row['experiment_A']})")
        mol = tasks.molecules(model)
        print(f"O2 singlet-triplet gap {mol['o2']['gap_eV']} eV, water IE vertical "
              f"{mol['water']['vertical_eV']} eV, adiabatic {mol['water']['adiabatic_eV']} eV")
        print("load", {k: round(v, 1) for k, v in model.load_s.items()})
    else:
        for row in tasks.timing(model):
            print(row)


if __name__ == "__main__":
    main()
