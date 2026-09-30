"""Run a visible ALCHEMI notebook cell inside the selected GPU SIF.

Load once with ``%run ../../scripts/notebook_alchemi.py``. The cell body is
ordinary Python, sent to ``python -`` in the container; no simulation code is
hidden in this adapter.
"""

import os
from pathlib import Path
import subprocess

from IPython import get_ipython


def alchemi(line: str, cell: str) -> None:
    fields = line.split()
    if len(fields) != 2 or any(not item.isdecimal() or int(item) < 1 for item in fields):
        raise ValueError("use %%alchemi REPLICAS STEPS with positive integers")
    model = Path(os.environ["MLIP_MODEL"]).resolve(strict=True)
    sif = Path(os.environ["MLIP_ALCHEMI_SIF"]).resolve(strict=True)
    if not model.is_file() or not sif.is_file():
        raise FileNotFoundError("MLIP_MODEL and MLIP_ALCHEMI_SIF must be files")
    command = [
        "apptainer", "exec", "--cleanenv", "--nv",
        "--env", "TORCH_COMPILE_DISABLE=1",
        "--env", "TORCH_DISABLE_NATIVE_JIT=1",
        "--env", f"MLIP_REPLICAS={fields[0]}",
        "--env", f"MLIP_STEPS={fields[1]}",
        "--bind", f"{model}:/models/mace.model:ro",
        str(sif), "python", "-",
    ]
    subprocess.run(command, input=cell, text=True, check=True)


shell = get_ipython()
if shell is None:
    raise RuntimeError("load this adapter in a Jupyter notebook")
shell.register_magic_function(alchemi, "cell", "alchemi")
