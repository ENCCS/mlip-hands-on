---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Build the ALCHEMI image

The Apptainer definition starts from an OCI image selected by digest. It
copies two lock files and the ALCHEMI MD example into the SIF. The lock files
fix the build dependencies and the runtime packages, including their hashes.
They are part of this repository, not generated during a class.

```{literalinclude} ../alchemi-aarch64.def
:language: text
:lines: 1-11
```

```{literalinclude} ../alchemi-aarch64.def
:language: text
:start-at: %post
:end-before: %labels
```

On an aarch64 builder with Apptainer, set a **new** output path outside Git
and run the supplied script. Do this before the GPU exercises; building an
image is not a notebook cell.

```bash
export MLIP_SIF_OUTPUT=/path/outside/git/alchemi-aarch64.sif
bash scripts/build-alchemi-sif.sh
```

The essential build steps are below. In the complete script, the
`apptainer build` line creates the image under a temporary `.partial` name.
Only after `apptainer sif list` succeeds is it moved to the final name.

```{literalinclude} ../scripts/build-alchemi-sif.sh
:language: bash
:start-at: staged=
:end-at: mv -n
:linenos:
:emphasize-lines: 5
```

The highlighted line is the `apptainer build` call. The numbered excerpt
shows the staging file and final move without the script's surrounding
preflight checks; use the complete script when building.

:::{note}
The MACE checkpoint stays outside the image. `scripts/run-alchemi.sh`
selects it explicitly and mounts it read-only. This keeps the image and
model identities independent and avoids rebuilding the SIF for each model.
:::

The base image digest, both lock files, and the resulting SIF identity all
matter. A successful `%test` confirms that the expected packages import; a
short GPU MD run in a real allocation is still needed before using a new
image in the exercise.
