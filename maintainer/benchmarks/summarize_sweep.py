#!/usr/bin/env python3
"""Validate the complete matched-NVE size/batch sweep, including every log."""
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
POINTS = [(2, 1), (4, 1), (10, 1), (16, 1), (2, 2), (4, 2), (2, 4), (4, 4)]
LOOP = re.compile(r"Loop time of ([0-9.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def finite_final_thermo(text, atoms):
    rows = [line.split() for line in text.splitlines()
            if re.match(r"^\s*210\s+\d+\s", line)]
    if (len(rows) != 1 or len(rows[0]) != 6 or int(rows[0][1]) != atoms or
            not all(math.isfinite(float(value)) for value in rows[0])):
        raise RuntimeError("missing finite final thermo")
    return float(rows[0][3])


def validate_case(root, row):
    round_no, atoms, replicas = (int(row[k]) for k in ("round", "atoms", "replicas"))
    method = row["method"]
    whole, md = (float(row[k]) for k in ("whole_seconds", "md_diagnostic_seconds"))
    if not all(math.isfinite(v) and v > 0 for v in (whole, md)) or md > whole:
        raise RuntimeError("invalid clock boundary")
    stem = f"round{round_no}-atoms{atoms}-r{replicas}-{method}"
    if method == "alchemi":
        records = [json.loads(line) for line in (root / f"{stem}.log").read_text().splitlines()
                   if line.startswith('{"state":')]
        if len(records) != 1:
            raise RuntimeError("missing unique ALCHEMI completion")
        record = records[0]
        if (record.get("state") != "completed" or record.get("ensemble") != "NVE" or
                record.get("atoms_per_replica") != atoms or record.get("replicas") != replicas or
                record.get("warmup_steps") != 10 or record.get("measured_steps") != 200 or
                len(record.get("final_potential_ev", [])) != replicas or
                not all(math.isfinite(v) for v in record["final_potential_ev"]) or
                not math.isclose(record["md_seconds"], md, rel_tol=1e-9)):
            raise RuntimeError("ALCHEMI result disagrees with workload")
    else:
        record = json.loads((root / f"{stem}-summary.json").read_text())
        if (record.get("state") != "completed" or record.get("ensemble") != "NVE" or
                record.get("atoms_per_replica") != atoms or record.get("replicas") != replicas or
                record.get("warmup_steps") != 10 or record.get("measured_steps") != 200 or
                len(record.get("replica_md_loop_seconds", [])) != replicas):
            raise RuntimeError("LAMMPS summary disagrees with workload")
        times = []
        for index in range(replicas):
            text = (root / f"{stem}-replicas" / f"replica-{index}.log").read_text()
            loops = LOOP.findall(text)
            if (text.count("Total wall time:") != 1 or "ERROR:" in text or
                    len(loops) != 3 or [int(v[2]) for v in loops] != [0, 10, 200] or
                    any(int(v[1]) != 1 or int(v[3]) != atoms for v in loops)):
                raise RuntimeError("LAMMPS replica incomplete")
            finite_final_thermo(text, atoms)
            seconds = float(loops[-1][0])
            if not math.isfinite(seconds) or seconds <= 0 or not math.isclose(
                    seconds, record["replica_md_loop_seconds"][index], rel_tol=1e-9):
                raise RuntimeError("LAMMPS replica timer mismatch")
            times.append(seconds)
        if not math.isclose(max(times), md, rel_tol=1e-9):
            raise RuntimeError("LAMMPS diagnostic clock mismatch")
        if method == "mps" and (root / f"{stem}-mps-proof").read_text().strip() != "observed":
            raise RuntimeError("MPS service not observed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "plain", "mps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {(r, c, n, m) for r in (1, 2, 3) for c, n in POINTS
                for m in ("alchemi", "ordinary", "mps")
                if (m != "mps" or n > 1) and
                (args.phase == "all" or (args.phase == "plain" and m != "mps") or
                 (args.phase == "mps" and m == "mps"))}
    with (args.results / "results.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise RuntimeError("unexpected matrix columns")
        rows = list(reader)
    if len(rows) != len(expected):
        raise RuntimeError("incomplete matrix")
    observed = set()
    for row in rows:
        key = (int(row["round"]), int(row["cells"]), int(row["replicas"]), row["method"])
        if (key not in expected or key in observed or int(row["atoms"]) != 8 * key[1]**3 or
                (int(row["warmup"]), int(row["steps"])) != (10, 200)):
            raise RuntimeError("duplicate or mismatched workload")
        observed.add(key)
        validate_case(args.results, row)
    groups = sorted({(int(row["atoms"]), int(row["replicas"]), row["method"]) for row in rows})
    medians = []
    for atoms, replicas, method in groups:
        group = [row for row in rows if (int(row["atoms"]), int(row["replicas"]), row["method"]) ==
                 (atoms, replicas, method)]
        whole = statistics.median(float(row["whole_seconds"]) for row in group)
        medians.append({"atoms": atoms, "replicas": replicas, "method": method,
                        "whole_seconds": whole, "replica_steps_per_whole_second": replicas * 200 / whole,
                        "md_diagnostic_seconds": statistics.median(float(row["md_diagnostic_seconds"]) for row in group)})
    result = {"state": "complete-pending-scientific-review", "phase": args.phase,
              "rows_checked": len(rows), "ensemble": "NVE", "warmup": 10, "steps": 200,
              "medians": medians}
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Checked {len(rows)} cases and every replica log")


if __name__ == "__main__":
    main()
