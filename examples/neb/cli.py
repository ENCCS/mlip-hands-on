"""Command-line entry: python cli.py {prefetch,versions,run,figures} [--model NAME]."""

from __future__ import annotations

import argparse
import json
import warnings

warnings.simplefilter("ignore")

import models

STEPS = ["prefetch", "versions", "run", "figures"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--model", choices=list(models.MODELS), help="model for run (default: all)")
    parser.add_argument("--images", type=int, default=7, help="moving images between end points")
    parser.add_argument("--fmax", type=float, default=0.05, help="force tolerance (eV/Å)")
    parser.add_argument("--optimiser", choices=["FIRE", "BFGS"], default="FIRE")
    args = parser.parse_args()
    names = [args.model] if args.model else list(models.MODELS)

    if args.step == "prefetch":
        for name in names:
            models.load(name)
            print("cached", name)
    elif args.step == "versions":
        import neb

        neb.RESULTS.mkdir(exist_ok=True)
        text = json.dumps(models.versions(), indent=2)
        (neb.RESULTS / "versions.json").write_text(text)
        print(text)
    elif args.step == "run":
        import neb

        for name in names:
            r = neb.run(name, images=args.images, fmax=args.fmax, optimiser=args.optimiser)
            print(f"{name}: barrier {r['barrier_eV']:.3f} eV, NEB {r['neb']['force_calls']} calls, "
                  f"{r['neb']['wall_s']} s, converged {r['neb']['converged']}")
    else:
        import figures

        figures.main()


if __name__ == "__main__":
    main()
