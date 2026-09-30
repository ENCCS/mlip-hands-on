"""Plot the NEB results into _static/neb-lifepo4.png and _static/neb-lifepo4-path.png."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms

from neb import RESULTS

FIGURES = RESULTS.parents[2] / "_static"
COLOURS = {"MACE-MP-0b": "#2a78d6", "Orb-v3": "#eb6834", "TensorNet": "#1baf7a"}
ELEMENT = {"Li": "#8a8984", "Fe": "#b5651d", "P": "#9b59b6", "O": "#e74c3c"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "font.size": 10, "figure.dpi": 150})


def load() -> dict:
    out = {}
    for name in COLOURS:
        path = RESULTS / f"{name.lower().replace('-', '_')}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, facecolor="white")
    plt.close(fig)
    print(FIGURES / name)


def profile(data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, d in data.items():
        c = COLOURS[name]
        ax1.plot(d["fit_path_A"], d["fit_energies_eV"], color=c,
                 label=f"{name}: {d['barrier_eV']:.2f} eV")
        ax1.plot(d["path_A"], d["energies_eV"], "o", color=c, ms=5)
        offset = np.linalg.norm(d["li_offset_xz_A"], axis=1)
        ax2.plot(d["path_A"], offset, "o-", color=c, label=name)
    ax1.set(xlabel="path length (Å)", ylabel="energy relative to start (eV)",
            title="CI-NEB, Li vacancy hop along [010]")
    ax1.legend(frameon=False)
    ax2.set(xlabel="path length (Å)", ylabel="offset from straight line (Å)",
            title="Curvature of the Li path")
    save(fig, "neb-lifepo4.png")


def path_view(name: str = "Orb-v3") -> None:
    """Relaxed initial supercell with every image of the hopping Li, seen down c and down a."""
    stem = name.lower().replace("-", "_")
    band = read(RESULTS / f"{stem}_band.extxyz", index=":")
    hopper = json.loads((RESULTS / f"{stem}.json").read_text())["hopper"]
    host = band[0].copy()
    del host[hopper]
    ghosts = Atoms("Li" * len(band), positions=[a.positions[hopper] for a in band], cell=host.cell)
    scene = host + ghosts
    centre = scene.cell.sum(axis=0) / 2 - ghosts.positions[len(band) // 2]
    scene.positions += centre
    scene.wrap()
    colours = [ELEMENT[s] for s in host.get_chemical_symbols()] + ["#111111"] * len(band)
    radii = [0.3 if s == "O" else 0.4 for s in host.get_chemical_symbols()] + [0.22] * len(band)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, rotation, title in zip(axes, ("0x,0y,0z", "-90y,0x,0z"), ("down c", "down a")):
        plot_atoms(scene, ax, rotation=rotation, colors=colours, radii=radii, show_unit_cell=2)
        ax.set(title=f"{name} path, view {title}", xticks=[], yticks=[])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in ELEMENT.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#111111", label="hopping Li, all images"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(FIGURES / "neb-lifepo4-path.png", facecolor="white")
    plt.close(fig)
    print(FIGURES / "neb-lifepo4-path.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    data = load()
    profile(data)
    path_view("Orb-v3" if "Orb-v3" in data else next(iter(data)))


if __name__ == "__main__":
    main()
