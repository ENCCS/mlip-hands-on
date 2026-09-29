#!/usr/bin/env bash
# Build the aarch64 ALCHEMI image from this lesson's definition file.
set -euo pipefail
lesson_root=$1
output=$2
cd "$lesson_root"
apptainer build "$output" alchemi-aarch64.def
