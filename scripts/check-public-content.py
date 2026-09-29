#!/usr/bin/env python3
"""Fail if tracked or untracked (non-ignored) lesson text contains known private site values.

This is a guardrail, not a substitute for reviewing the rendered HTML.
Only file names, line numbers, and finding categories are printed.
"""

from pathlib import Path
import re
import subprocess
import sys


TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".txt", ".def", ".yml", ".yaml"}
PRIVATE_PATTERNS = {
    "personal home path": re.compile(r"/(?:home/" + "wei|home/liwei|Users/liwei|Users/karim)/"),
    "Leonardo scratch path": re.compile(r"/leonardo_" + r"(?:scratch|work)/[a-z]+/(?!<)[^/\s]+"),
    "Cineca project code": re.compile(r"\bIscr" + r"[ABC]_[A-Za-z0-9]+"),
    "Leonardo node hostname": re.compile(r"\b(?:login" + r"\d+(?:-ext)?\.leonardo|lrdn\d{4})\b"),
    "Arrhenius project identifier": re.compile(r"/nobackup/proj/disk/snic" + r"\d{4}-\d{2}-\d/"),
    "LUMI project identifier": re.compile(r"/flash/project_" + r"\d+/"),
    "JUPITER project identifier": re.compile(r"/e/project1/e-dev-" + r"\d{4}[a-z]\d{2}-\d+/"),
    "allocation account": re.compile(r"(?:naiss" + r"\d{4}-\d{2}-\d|EUHPC_D\d{2}_\d{3})"),
    "GitHub token": re.compile(r"(?:gh" + r"[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "URL token": re.compile(r"[?&]token=" + r"[^\s&#'\"]{12,}"),
}


def main() -> int:
    # Tracked files plus untracked, non-ignored files, so new work is checked
    # before it is committed.
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    ).split(b"\0")
    findings = 0
    for raw_name in names:
        if not raw_name:
            continue
        path = Path(raw_name.decode("utf-8"))
        if path.suffix not in TEXT_SUFFIXES and path.name != ".env.example":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for category, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(line):
                    print(f"{path}:{number}: {category}", file=sys.stderr)
                    findings += 1
    config = Path("conf.py").read_text(encoding="utf-8")
    if not re.search(r"""(?m)^\s*nb_execution_mode\s*=\s*["']off["']\s*$""", config):
        print("conf.py: notebook execution must default to off", file=sys.stderr)
        findings += 1
    overrides = set()
    for raw_name in names:
        if not raw_name:
            continue
        path = Path(raw_name.decode("utf-8"))
        if path.suffix != ".md":
            continue
        source = path.read_text(encoding="utf-8")
        if source.startswith("---\n"):
            header = source.split("\n---\n", 1)[0]
            mode = re.search(r"(?m)^\s*execution_mode:\s*([^\s#]+)", header)
            if mode and mode.group(1).strip("\"'") != "off":
                overrides.add(path.as_posix())
    if overrides != {"episodes/07-reviewed-results.md"}:
        print("Only the reviewed offline-results page may execute during Sphinx builds", file=sys.stderr)
        findings += 1
    if findings:
        print(f"Found {findings} publication check finding(s)", file=sys.stderr)
        return 1
    print("No known private site values in lesson text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
