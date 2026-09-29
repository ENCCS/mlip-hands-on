"""Plot the saved JSON/CSV results into _static/matgl-*.png."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, RESULTS, read_json

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
COLOURS = {"TensorNet": BLUE, "M3GNet": ORANGE, "finetune": BLUE, "scratch": ORANGE}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "font.size": 10, "figure.dpi": 150})


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"matgl-{name}.png", facecolor="white")
    plt.close(fig)
    print(FIGURES / f"matgl-{name}.png")


def relax_md() -> None:
    d = read_json("relax_md.json")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.3))
    for label, m in d["models"].items():
        c = COLOURS[label]
        ax1.plot(m["relax"]["energies"], color=c, label=label)
        t = np.arange(len(m["md"]["temperature"])) * m["md"]["interval"] / 1000
        ax2.plot(t, m["md"]["temperature"], color=c, label=label)
        ax3.plot(t, m["md"]["energy_per_atom"], color=c, label=label)
    ax1.set(xlabel="optimiser step", ylabel="energy (eV/cell)", title="CsCl relaxation")
    ax2.axhline(d["temperature_K"], color=GREY, lw=1, ls="--")
    ax2.set(xlabel="time (ps)", ylabel="temperature (K)", title="NVT Langevin, 54 atoms")
    ax3.set(xlabel="time (ps)", ylabel="potential energy (eV/atom)", title="MD energy")
    ax1.legend(frameon=False)
    save(fig, "relax-md")


def lattice() -> None:
    df = pd.read_csv(RESULTS / "lattice.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 2]})
    lim = [df.a_expt.min() - 0.2, df.a_expt.max() + 0.3]
    ax1.plot(lim, lim, color=GREY, lw=1)
    x = np.arange(len(df))
    for i, label in enumerate(["TensorNet", "M3GNet"]):
        ax1.scatter(df.a_expt, df[label], s=22, color=COLOURS[label], label=label, zorder=3)
        ax2.bar(x + (i - 0.5) * 0.4, df[f"{label}_pct"], 0.38, color=COLOURS[label], label=label)
    ax1.set(xlim=lim, ylim=lim, xlabel="experimental a (Å)", ylabel="predicted a (Å)")
    ax1.legend(frameon=False)
    ax2.axhline(0, color="#52514e", lw=0.8)
    ax2.set_xticks(x, df.formula, rotation=60, fontsize=8)
    ax2.set(ylabel="error vs experiment (%)")
    save(fig, "lattice")


def predict() -> None:
    d = read_json("predict.json")["crystals"]
    names = list(d)
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    series = [("unrelaxed", ORANGE), ("relaxed", BLUE)]
    for i, (key, c) in enumerate(series):
        ax1.bar(x + (i - 1) * 0.27, [d[n][key]["eform_eV_atom"] for n in names], 0.25, color=c, label=key)
    ax1.bar(x + 0.27, [d[n]["eform_expt_eV_atom"] for n in names], 0.25, color=GREY, label="experiment")
    ax1.set(ylabel="formation energy (eV/atom)", title="M3GNet-Eform-MP-2019.4.1")
    for i, (key, c) in enumerate([("gap_PBE_eV", BLUE), ("gap_GLLB-SC_eV", AQUA)]):
        ax2.bar(x + (i - 1) * 0.27, [d[n]["relaxed"][key] for n in names], 0.25, color=c,
                label=key[4:-3] + " (relaxed)")
    ax2.bar(x + 0.27, [d[n]["gap_expt_eV"] for n in names], 0.25, color=GREY, label="experiment")
    ax2.set(ylabel="band gap (eV)", title="MEGNet-BandGap-mfi-MP-2019.4.1")
    for ax in (ax1, ax2):
        ax.set_xticks(x, names)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    save(fig, "predict")


def train() -> None:
    d = read_json("train.json")["runs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = {"finetune": "fine-tuned MatPES M3GNet", "scratch": "M3GNet from scratch"}
    for key, run in d.items():
        c, cur = COLOURS[key], run["curve"]
        ax1.plot(cur["train_Total_Loss"], color=c, label=f"{labels[key]}, train")
        ax1.plot(cur["val_Total_Loss"], color=c, ls="--", label=f"{labels[key]}, validation")
        ax2.plot(cur["val_Force_MAE"], color=c, label=labels[key])
    ax1.set(yscale="log", xlabel="epoch", ylabel="loss (energy + force MSE)")
    ax2.set(yscale="log", xlabel="epoch", ylabel="validation force MAE (eV/Å)")
    ax1.legend(frameon=False, fontsize=8)
    save(fig, "train")


def main() -> None:
    for plot in (relax_md, lattice, predict, train):
        plot()
