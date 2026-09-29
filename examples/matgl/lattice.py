"""Relax cubic crystals from a common starting cell and compare with experiment."""

from __future__ import annotations

import pandas as pd
from matgl.ext.ase import Relaxer

from common import HERE, PES_MODELS, RESULTS, device, load, save_json, timer
from prototypes import build

A_START = 5.0


def reference() -> pd.DataFrame:
    return pd.read_csv(HERE / "lattice_reference.csv", comment="#")


def relaxed_a(relaxer: Relaxer, row) -> float:
    structure = build(row.prototype, row.species.split(), A_START)
    final = relaxer.relax(structure, fmax=0.01, steps=1000)["final_structure"]
    return final.volume ** (1 / 3)  # conventional cubic cell


def summarise(df: pd.DataFrame, label: str) -> dict:
    err = df[label] - df["a_expt"]
    pct = 100 * err / df["a_expt"]
    return {"mae_A": round(float(err.abs().mean()), 4), "mean_signed_pct": round(float(pct.mean()), 2),
            "mape_pct": round(float(pct.abs().mean()), 2), "max_abs_pct": round(float(pct.abs().max()), 2)}


def main() -> dict:
    df = reference()
    out = {"a_start_A": A_START, "n_crystals": len(df), "models": {}}
    for label, name in PES_MODELS.items():
        times: dict = {}
        relaxer = Relaxer(potential=load(name, device()))
        with timer(times, "total_s"):
            df[label] = [round(relaxed_a(relaxer, r), 4) for r in df.itertuples()]
        df[f"{label}_pct"] = (100 * (df[label] - df["a_expt"]) / df["a_expt"]).round(2)
        out["models"][label] = {"name": name, **summarise(df, label), **times}
        print(label, out["models"][label])
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "lattice.csv", index=False)
    save_json(out, "lattice.json")
    print(df.to_string(index=False))
    return out
