#!/usr/bin/env python3
"""Keep the visible one-cell MD source identical to the CLI Python file."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
page = (ROOT / "content/episodes/04-silicon-md.md").read_text(encoding="utf-8")
marker = "```{code-cell} ipython3\n%%alchemi 1 200\n"
if page.count(marker) != 1:
    raise SystemExit("expected exactly one ALCHEMI MD cell")
cell = page.split(marker, 1)[1].split("\n```", 1)[0] + "\n"
source = (ROOT / "examples/alchemi_si_one_cell.py").read_text(encoding="utf-8")
if cell != source:
    raise SystemExit("ALCHEMI notebook cell differs from examples/alchemi_si_one_cell.py")
print("ALCHEMI notebook cell matches the CLI Python source")
