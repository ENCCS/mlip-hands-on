#!/usr/bin/env python3
"""Accept only a complete matched-NVE eight-trajectory timing matrix."""

import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path


FIELDS = ["round", "cells", "atoms", "method", "replicas", "warmup",
          "steps", "whole_seconds", "md_diagnostic_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "plain", "mps"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = args.results / "results.tsv"
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise RuntimeError("matrix columns do not match the contract")
        rows = list(reader)
    methods = ("alchemi", "ordinary", "mps") if args.phase == "all" else (
        ("alchemi", "ordinary") if args.phase == "plain" else ("mps",))
    expected = {(round_no, atoms, method)
                for round_no in (1, 2, 3) for atoms in (64, 512)
                for method in methods}
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
        if (not all(math.isfinite(value) and value > 0 for value in (whole, md_diagnostic)) or
                md_diagnostic > whole):
            raise RuntimeError("invalid timing")
        legacy_stem = f"round{round_no}-atoms{atoms}-{method}"
        current_stem = f"round{round_no}-atoms{atoms}-r8-{method}"
        candidates = [stem for stem in (legacy_stem, current_stem)
                      if (args.results / f"{stem}.log").is_file()]
        if len(candidates) != 1:
            raise RuntimeError("missing or ambiguous case log")
        stem = candidates[0]
        if method == "alchemi":
            lines = (args.results / f"{stem}.log").read_text().splitlines()
            records = [json.loads(line) for line in lines if line.startswith('{"state":')]
            if len(records) != 1:
                raise RuntimeError(f"missing ALCHEMI completion for {stem}")
            record = records[0]
            if (record["state"] != "completed" or record.get("ensemble") != "NVE" or record["replicas"] != 8 or
                    record["atoms_per_replica"] != atoms or
                    record["warmup_steps"] != 10 or record["measured_steps"] != 200 or
                    len(record["final_potential_ev"]) != 8 or
                    not all(math.isfinite(value) for value in record["final_potential_ev"]) or
                    not math.isclose(record["md_seconds"], md_diagnostic, rel_tol=1e-9)):
                raise RuntimeError(f"invalid ALCHEMI completion for {stem}")
        else:
            record = json.loads((args.results / f"{stem}-summary.json").read_text())
            if (record["state"] != "completed" or record.get("ensemble") != "NVE" or record["replicas"] != 8 or
                    record["atoms_per_replica"] != atoms or
                    record["warmup_steps"] != 10 or record["measured_steps"] != 200 or
                    len(record["replica_md_loop_seconds"]) != 8 or
                    not math.isclose(record["longest_replica_md_loop_seconds"],
                                     md_diagnostic, rel_tol=1e-9)):
                raise RuntimeError(f"invalid LAMMPS completion for {stem}")
            for replica in range(8):
                text = (args.results / f"{stem}-replicas" / f"replica-{replica}.log").read_text()
                loops = re.findall(r"Loop time of ([0-9.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms", text)
                if (text.count("Total wall time:") != 1 or "ERROR:" in text or
                        len(loops) != 3 or [int(row[2]) for row in loops] != [0, 10, 200] or
                        any(int(row[1]) != 1 or int(row[3]) != atoms for row in loops) or
                        not math.isclose(float(loops[-1][0]), record["replica_md_loop_seconds"][replica], rel_tol=1e-9)):
                    raise RuntimeError(f"replica log disagrees with summary for {stem}")
                final_rows = [line.split() for line in text.splitlines()
                              if re.match(r"^\s*210\s+\d+\s", line)]
                if len(final_rows) != 1 or len(final_rows[0]) != 6 or not all(
                        math.isfinite(float(value)) for value in final_rows[0]):
                    raise RuntimeError(f"missing finite final thermo for {stem}")
        normalized.append({"round": round_no, "atoms": atoms, "method": method,
                           "whole_seconds": whole,
                           "md_diagnostic_seconds": md_diagnostic})
    if observed != expected:
        raise RuntimeError("matrix missing a case")
    medians = {str(atoms): {
        method: statistics.median(row["whole_seconds"] for row in normalized
                                  if row["atoms"] == atoms and row["method"] == method)
        for method in methods}
        for atoms in (64, 512)}
    result = {"state": "complete-pending-scientific-review", "format": 1,
              "phase": args.phase, "ensemble": "NVE", "replicas": 8, "warmup_steps": 10,
              "measured_steps": 200, "whole_workflow_medians": medians,
              "clock_note": "MD diagnostics are not directly comparable across engines",
              "rows": normalized}
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print(f"Validated {len(expected)} completed cases; scientific review remains required")


if __name__ == "__main__":
    main()
