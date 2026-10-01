#!/usr/bin/env python3
"""Validate three complete rounds of matched NVE strong/weak scaling."""
import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from summarize_sweep import LOOP, finite_final_thermo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.results / "results.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["round", "scaling", "ranks", "atoms", "warmup", "steps", "md_seconds"]:
            raise RuntimeError("unexpected scaling columns")
        rows = list(reader)
    expected = {(r, mode, n) for r in (1, 2, 3) for mode in ("strong", "weak") for n in (1, 2, 4)}
    if len(rows) != 18:
        raise RuntimeError("incomplete scaling matrix")
    observed = set()
    for row in rows:
        round_no, mode, ranks = int(row["round"]), row["scaling"], int(row["ranks"])
        key = (round_no, mode, ranks)
        atoms = 32768 if mode == "strong" else {1: 8000, 2: 17576, 4: 32768}.get(ranks)
        md = float(row["md_seconds"])
        if (key not in expected or key in observed or int(row["atoms"]) != atoms or
                (int(row["warmup"]), int(row["steps"])) != (10, 200) or
                not math.isfinite(md) or md <= 0):
            raise RuntimeError("scaling workload mismatch")
        observed.add(key)
        stem = f"round{round_no}-{mode}-g{ranks}-a{atoms}"
        text = (args.results / f"{stem}.log").read_text()
        loops = LOOP.findall(text)
        if (text.count("Total wall time:") != 1 or "ERROR:" in text or len(loops) != 3 or
                re.findall(r"will use up to (\d+) GPU\(s\) per node", text) != [str(ranks)] or
                [int(v[2]) for v in loops] != [0, 10, 200] or
                any(int(v[1]) != ranks or int(v[3]) != atoms for v in loops) or
                not math.isclose(float(loops[-1][0]), md, rel_tol=1e-9)):
            raise RuntimeError("scaling log mismatch")
        finite_final_thermo(text, atoms)
        result = json.loads((args.results / f"{stem}-summary.json").read_text())
        if (result.get("state") != "completed" or result.get("ranks") != ranks or
                result.get("atoms") != atoms or result.get("ensemble") != "NVE" or
                result.get("warmup_steps") != 10 or result.get("measured_steps") != 200 or
                not math.isclose(result["md_seconds"], md, rel_tol=1e-9)):
            raise RuntimeError("scaling summary mismatch")
    medians = [{"scaling": mode, "gpus": n, "atoms": 32768 if mode == "strong" else
                {1: 8000, 2: 17576, 4: 32768}[n],
                "md_seconds": statistics.median(float(row["md_seconds"]) for row in rows
                                                if row["scaling"] == mode and int(row["ranks"]) == n)}
               for mode in ("strong", "weak") for n in (1, 2, 4)]
    result = {"state": "complete-pending-scientific-review", "rows_checked": 18,
              "ensemble": "NVE", "warmup": 10, "steps": 200, "medians": medians}
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("Checked all 18 scaling logs and summaries")


if __name__ == "__main__":
    main()
