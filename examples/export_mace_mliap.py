#!/usr/bin/env python3
"""Export the pinned MACE checkpoint for LAMMPS ML-IAP/Kokkos."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch


MODEL_SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.model.is_file() or sha256(args.model) != MODEL_SHA256:
        parser.error("the original checkpoint is absent or has the wrong SHA-256")
    if args.output.exists():
        parser.error("the output already exists; select a fresh path")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("exactly one visible CUDA GPU is required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "model.model"
        shutil.copyfile(args.model, staged)
        result = subprocess.run(
            [sys.executable, "-m", "mace.cli.create_lammps_model", str(staged),
             "--format", "mliap", "--dtype", "float32"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if result.returncode:
            detail = result.stderr[-800:].decode("utf-8", errors="replace")
            raise RuntimeError(f"MACE ML-IAP export failed (exit {result.returncode}): {detail}")
        exported = Path(str(staged) + "-mliap_lammps.pt")
        if not exported.is_file() or not exported.stat().st_size:
            raise RuntimeError("MACE did not create the ML-IAP export")
        digest = sha256(exported)
        os.link(exported, args.output)  # Fails if another process created the output.
    print(f"ML-IAP export: {args.output.name}, SHA-256 {digest}")


if __name__ == "__main__":
    main()
