"""Command-line entry: python cli.py {prefetch,versions,relax-md,lattice,predict,train,figures}."""

from __future__ import annotations

import argparse

import common

STEPS = ["prefetch", "versions", "relax-md", "lattice", "predict", "train", "figures"]


def prefetch() -> None:
    """Download all pretrained models (needs internet, so run it on a login node)."""
    import matgl

    names = matgl.get_available_pretrained_models()
    print("\n".join(names))
    for name in [*common.PES_MODELS.values(), common.EFORM_MODEL, common.GAP_MODEL]:
        common.load(name)
        print("cached", name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--steps", type=int, default=1000, help="MD steps for relax-md")
    parser.add_argument("--epochs", type=int, default=50, help="epochs for train")
    parser.add_argument("--float-bits", type=int, choices=(32, 64), default=common.FLOAT_BITS,
                        help="float size (default: $MATGL_FLOAT_BITS or 32)")
    args = parser.parse_args()
    common.set_precision(args.float_bits)

    if args.step == "prefetch":
        prefetch()
    elif args.step == "versions":
        print(common.save_json(common.versions(), "versions.json").read_text())
    elif args.step == "relax-md":
        import relax_md
        relax_md.main(steps=args.steps)
    elif args.step == "lattice":
        import lattice
        lattice.main()
    elif args.step == "predict":
        import predict
        predict.main()
    elif args.step == "train":
        import train
        train.main(epochs=args.epochs)
    else:
        import figures
        figures.main()


if __name__ == "__main__":
    main()
