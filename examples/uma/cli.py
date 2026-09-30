"""Command-line entry: python cli.py {prefetch,versions,run,timing}."""

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
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()
    tasks.RESULTS.mkdir(exist_ok=True)

    if args.step == "prefetch":
        models.UMA(args.dtype)
        print("cached", models.CHECKPOINT)
    elif args.step == "versions":
        text = json.dumps(models.versions(), indent=2)
        (tasks.RESULTS / "versions.json").write_text(text + "\n")
        print(text)
    elif args.step == "run":
        uma = models.UMA(args.dtype)
        print(f"loaded {models.CHECKPOINT} ({args.dtype}) in {uma.load_s:.1f} s")
        for row in tasks.lattice(uma):
            print(f"{row['crystal']}: a = {row['uma_A']} Å (PBE {row['pbe_A']}, experiment {row['experiment_A']})")
        mol = tasks.molecules(uma)
        print(f"O2 singlet-triplet gap {mol['o2']['gap_eV']} eV, water IE vertical "
              f"{mol['water']['vertical_eV']} eV, adiabatic {mol['water']['adiabatic_eV']} eV")
    else:
        rows = tasks.timing(models.UMA(args.dtype))
        tasks.save_csv(f"timing_{args.dtype}.csv", rows)
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
