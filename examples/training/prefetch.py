"""Download the datasets and the pretrained model, and write small subsets.

Run once on a login node; the training jobs then need no network access.
"""

import json
import random
import shutil
from pathlib import Path

from ase.stress import voigt_6_to_full_3x3_stress
from huggingface_hub import hf_hub_download

from . import config

PES_KEYS = ("structure", "energy", "forces", "matpes_id", "formula_pretty")


def _dataset_file(repo: str, name: str, data: Path) -> Path:
    return Path(hf_hub_download(repo, name, repo_type="dataset", cache_dir=data / "hf"))


def eform_subset(data: Path, n: int, seed: int) -> int:
    with open(_dataset_file(config.EFORM_REPO, config.EFORM_FILE, data)) as f:
        rows = json.load(f)
    random.Random(seed).shuffle(rows)
    keep = [
        {"material_id": r["material_id"], "structure": r["structure"], "eform": r["formation_energy_per_atom"]}
        for r in rows[:n]
    ]
    (data / config.EFORM_SUBSET).write_text(json.dumps(keep))
    return len(keep)


def full_stress(voigt: list[float]) -> list[list[float]]:
    """MatPES stores Voigt order (xx, yy, zz, yz, xz, xy); MatGL training compares 3x3 tensors."""
    return voigt_6_to_full_3x3_stress(voigt).tolist()


def pes_subset(data: Path, element: str, max_sites: int, n: int, seed: int) -> int:
    pool = _dataset_file(config.MATPES_REPO, config.MATPES_POOL, data)
    with open(pool) as f:
        rows = [r for r in map(json.loads, f) if element in r["elements"] and r["nsites"] <= max_sites]
    random.Random(seed).shuffle(rows)
    with open(config.pes_subset(data, element), "w") as f:
        for r in rows[:n]:
            f.write(json.dumps({**{k: r[k] for k in PES_KEYS}, "stress": full_stress(r["stress"])}) + "\n")
    atoms = _dataset_file(config.MATPES_REPO, config.MATPES_ATOMS, data)
    shutil.copyfile(atoms, config.atom_refs(data))
    return min(n, len(rows))


def pretrained(data: Path) -> Path:
    target = config.model_dir(data)
    for name in config.MODEL_FILES:
        hf_hub_download(f"materialyze/{config.PRETRAINED}", name, local_dir=target)
    return target


def run(args) -> None:
    args.data.mkdir(parents=True, exist_ok=True)
    print("pretrained model:", pretrained(args.data).name)
    n = pes_subset(args.data, args.element, args.max_sites, args.n_pes, args.seed)
    print(f"r2SCAN structures containing {args.element}: {n}")
    if not args.skip_eform:
        print("formation-energy structures:", eform_subset(args.data, args.n_eform, args.seed))
