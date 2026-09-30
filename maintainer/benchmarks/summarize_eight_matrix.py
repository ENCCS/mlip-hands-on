#!/usr/bin/env python3
"""Accept only a complete matched-NVE eight-trajectory timing matrix."""

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path


FIELDS = ["round", "cells", "atoms", "method", "replicas", "warmup",
          "steps", "whole_seconds", "md_diagnostic_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = args.results / "results.tsv"
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise RuntimeError("matrix columns do not match the contract")
        rows = list(reader)
    expected = {(round_no, atoms, method)
                for round_no in (1, 2, 3) for atoms in (64, 512)
                for method in ("alchemi", "ordinary", "mps")}
    if len(rows) != len(expected):
        raise RuntimeError("matrix incomplete or duplicated")
    observed = set()
    normalized = []
    for row in rows:
        round_no, cells, atoms = (int(row[name]) for name in ("round", "cells", "atoms"))
        method = row["method"]
        key = (round_no, atoms, method)
        if (key not in expected or key in observed or cells not in (2, 4) or
                atoms != 8 * cells**3 or
                (int(row["replicas"]), int(row["warmup"]), int(row["steps"])) !=
                (8, 10, 200)):
            raise RuntimeError("matrix workload or identity mismatch")
        observed.add(key)
        whole = float(row["whole_seconds"])
        md_diagnostic = float(row["md_diagnostic_seconds"])
        if not all(math.isfinite(value) and value > 0 for value in (whole, md_diagnostic)):
            raise RuntimeError("invalid timing")
        stem = f"round{round_no}-atoms{atoms}-{method}"
        if method == "alchemi":
            lines = (args.results / f"{stem}.log").read_text().splitlines()
            records = [json.loads(line) for line in lines if line.startswith('{"state":')]
            if len(records) != 1:
                raise RuntimeError(f"missing ALCHEMI completion for {stem}")
            record = records[0]
            if (record["state"] != "completed" or record["replicas"] != 8 or
                    record["atoms_per_replica"] != atoms or
                    record["warmup_steps"] != 10 or record["measured_steps"] != 200 or
                    len(record["final_potential_ev"]) != 8 or
                    not math.isclose(record["md_seconds"], md_diagnostic, rel_tol=1e-9)):
                raise RuntimeError(f"invalid ALCHEMI completion for {stem}")
        else:
            record = json.loads((args.results / f"{stem}-summary.json").read_text())
            if (record["state"] != "completed" or record["replicas"] != 8 or
                    record["atoms_per_replica"] != atoms or
                    record["warmup_steps"] != 10 or record["measured_steps"] != 200 or
                    len(record["replica_md_loop_seconds"]) != 8 or
                    not math.isclose(record["longest_replica_md_loop_seconds"],
                                     md_diagnostic, rel_tol=1e-9)):
                raise RuntimeError(f"invalid LAMMPS completion for {stem}")
        normalized.append({"round": round_no, "atoms": atoms, "method": method,
                           "whole_seconds": whole,
                           "md_diagnostic_seconds": md_diagnostic})
    if observed != expected:
        raise RuntimeError("matrix missing a case")
    medians = {str(atoms): {
        method: statistics.median(row["whole_seconds"] for row in normalized
                                  if row["atoms"] == atoms and row["method"] == method)
        for method in ("alchemi", "ordinary", "mps")}
        for atoms in (64, 512)}
    result = {"state": "complete-pending-scientific-review", "format": 1,
              "ensemble": "NVE", "replicas": 8, "warmup_steps": 10,
              "measured_steps": 200, "whole_workflow_medians": medians,
              "clock_note": "MD diagnostics are not directly comparable across engines",
              "rows": normalized}
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print("Validated 18 completed cases; scientific review remains required")


if __name__ == "__main__":
    main()
