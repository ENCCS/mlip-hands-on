#!/usr/bin/env python3
"""Execute selected actual MyST notebooks in an allocated GPU job.

This emits only a bounded outcome; model paths, outputs and scheduler logs
remain in the site's private job directory. It does not submit a job.
"""

import os
from pathlib import Path

import jupytext
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "04-silicon-md",
    "05-batched-md",
    "06-lammps-replicas",
    "07-relaxation",
}

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("run only inside an allocated Slurm GPU job")
for name in ("MLIP_MODEL", "MLIP_MLIAP_MODEL", "MLIP_ALCHEMI_SIF",
             "MLIP_NATIVE_PREFIX", "MLIP_NATIVE_PYTHON", "MLIP_SITE"):
    if not os.environ.get(name):
        raise SystemExit(f"missing {name}")

chosen = os.environ.get("MLIP_NOTEBOOK_PAGES", "04-silicon-md").split(",")
if not chosen or any(name not in PAGES for name in chosen) or len(chosen) != len(set(chosen)):
    raise SystemExit("MLIP_NOTEBOOK_PAGES must name distinct supported episode basenames")

for name in chosen:
    page = ROOT / "content/episodes" / f"{name}.md"
    notebook = jupytext.read(page, fmt="md:myst")
    client = NotebookClient(
        notebook,
        timeout=1800,
        resources={"metadata": {"path": str(page.parent)}},
    )
    executed = client.execute()
    output_text = "\n".join(
        output.get("text", "")
        for cell in executed.cells if cell.cell_type == "code"
        for output in cell.get("outputs", []) if output.output_type == "stream"
    )
    if name == "04-silicon-md":
        if "Completed 1 trajectories of 64 Si atoms for 200 steps" not in output_text:
            raise SystemExit("ALCHEMI notebook cell did not complete")
        if "Total wall time:" not in output_text:
            raise SystemExit("LAMMPS notebook cell did not complete")
    if name == "06-lammps-replicas" and "Eight LAMMPS process logs:" not in output_text:
        raise SystemExit("eight-process notebook cell did not complete")
    print(f"MyST notebook {name}: GPU cells completed", flush=True)
