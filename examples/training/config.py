"""Names of the datasets, the pretrained model and the files derived from them."""

from pathlib import Path

PRETRAINED = "TensorNet-PES-MatPES-PBE-2025.2"
MODEL_FILES = ("model.pt", "state.pt", "model.json")

EFORM_REPO = "materialyze/mp.eform.2018.6.1"
EFORM_FILE = "mp.eform.2018.6.1.json"
EFORM_SUBSET = "mp_eform_subset.json"

MATPES_REPO = "materialyze/matpes"
# The PBE model never saw the test split, so it is a clean pool for r2SCAN data.
MATPES_POOL = "MatPES-R2SCAN-2025.2-test.jsonl"
MATPES_ATOMS = "MatPES-R2SCAN-atoms.jsonl"


def pes_subset(data: Path, element: str) -> Path:
    return data / f"matpes_r2scan_{element}.jsonl"


def atom_refs(data: Path) -> Path:
    return data / MATPES_ATOMS


def model_dir(data: Path) -> Path:
    return data / "models" / PRETRAINED
