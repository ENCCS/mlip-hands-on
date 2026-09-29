"""Summarise timings and check the batch against the serial baseline."""

import hashlib
import json
from pathlib import Path

from ase.io import write

from .runners import RelaxResult


def sha256(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarise(config: dict, labels: list[str], serial: RelaxResult,
              batched: RelaxResult, checkpoint: str | None) -> dict:
    n_base, n_total = len(serial.energies), len(labels)
    estimate = serial.wall_s / n_base * n_total
    diffs = [abs(a - b) for a, b in zip(serial.energies, batched.energies)]
    return {
        **config,
        "n_structures": n_total,
        "baseline_n": n_base,
        "baseline_wall_s": serial.wall_s,
        "baseline_est_full_s": estimate,
        "batched_wall_s": batched.wall_s,
        "speedup_vs_estimated_serial": estimate / batched.wall_s,
        "max_abs_energy_diff_eV": max(diffs),
        "checkpoint_sha256": sha256(checkpoint),
        "energies_eV": dict(zip(labels, batched.energies)),
    }


def print_summary(s: dict) -> None:
    n = s["n_structures"]
    print(f"{'':26}{'wall time':>12}{'s/structure':>14}")
    print(f"{'serial, measured':26}{s['baseline_wall_s']:>10.1f} s"
          f"{s['baseline_wall_s'] / s['baseline_n']:>12.2f} s   ({s['baseline_n']} structures)")
    print(f"{'serial, estimated':26}{s['baseline_est_full_s']:>10.1f} s"
          f"{s['baseline_est_full_s'] / n:>12.2f} s   ({n} structures)")
    print(f"{'batched, measured':26}{s['batched_wall_s']:>10.1f} s"
          f"{s['batched_wall_s'] / n:>12.2f} s   ({n} structures)")
    print(f"estimated serial / batched: {s['speedup_vs_estimated_serial']:.2f}")
    print(f"max |E_ASE - E_TorchSim| on the serial subset: "
          f"{s['max_abs_energy_diff_eV']:.2e} eV")


def write_outputs(outdir: str, summary: dict, relaxed: list) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "relaxed.extxyz", relaxed)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out
