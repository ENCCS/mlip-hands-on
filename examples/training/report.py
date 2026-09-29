"""JSON records and figures for the training examples."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"


def environment() -> dict[str, str]:
    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu"
    return {"device": device, **{p: version(p) for p in ("matgl", "torch", "lightning", "torch_geometric")}}


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")


def eform_figure(curve: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for key, colour, label in (("train_MAE", BLUE, "training"), ("val_MAE", ORANGE, "validation")):
        ax.plot(range(1, len(curve[key]) + 1), curve[key], color=colour, lw=2, label=label)
    ax.set(xlabel="epoch", ylabel="MAE (eV/atom)", yscale="log")
    _plain(ax.yaxis)
    _finish(fig, ax, path)


def finetune_figure(cases: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    panels = (("energy_mae_mev_atom", "energy MAE (meV/atom)"), ("force_mae_mev_A", "force MAE (meV/Å)"))
    for ax, (key, label) in zip(axes, panels):
        for kind, colour, name in (("fine-tuned", BLUE, "fine-tuned"), ("scratch", ORANGE, "from scratch")):
            rows = [c for c in cases if c["kind"] == kind]
            ax.plot([c["n_train"] for c in rows], [c[key] for c in rows], "o-", color=colour, lw=2, ms=8, label=name)
        for c, style in zip([c for c in cases if c["kind"] == "zero-shot"], ("--", ":")):
            ax.axhline(c[key], color=GREY, ls=style, lw=1.5, label=c["name"])
        ax.set(xlabel="training structures", ylabel=label, xscale="log", yscale="log")
        ax.set_xticks(sorted({c["n_train"] for c in cases if c["n_train"]}))
        _plain(ax.xaxis)
        _plain(ax.yaxis)
    _finish(fig, axes[1], path)


def _plain(axis) -> None:
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def _finish(fig, ax, path: Path) -> None:
    ax.legend(frameon=False, fontsize=8)
    for a in fig.axes:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
