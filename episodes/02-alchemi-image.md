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

:::{dropdown} alchemi-aarch64.def
```{literalinclude} ../alchemi-aarch64.def
:language: text
:lines: 1-11
:lineno-match:
```
:::

:::{dropdown} alchemi-aarch64.def
```{literalinclude} ../alchemi-aarch64.def
:language: text
:start-at: %post
:end-before: %labels
:lineno-match:
```
:::

The complete [Apptainer definition](../alchemi-aarch64.def) and its
[build lock](../locks/build-requirements.lock) and
[runtime lock](../locks/requirements.lock) are in the checkout. The
definition installs the locked packages; the locks are not model weights.

On an aarch64 builder with Apptainer, set a **new** output path outside Git
and run the supplied script. Do this before the GPU exercises; building an
image is not a notebook cell.

```bash
export MLIP_SIF_OUTPUT=/path/outside/git/alchemi-aarch64.sif
bash scripts/build-alchemi-sif.sh
```

The essential build steps are below. The script first builds into a new
temporary `.partial` file (called `staged` in the shell code). That prevents
an incomplete build from appearing at the final SIF path. After
`apptainer sif list` succeeds, it moves the finished file into place.

:::{dropdown} build-alchemi-sif.sh
```{literalinclude} ../scripts/build-alchemi-sif.sh
:language: bash
:start-at: staged=
:end-at: mv -n
:lineno-match:
:emphasize-lines: 5
```
:::

The highlighted line is the `apptainer build` call. This excerpt omits
the preflight checks: run the complete [build script](../scripts/build-alchemi-sif.sh),
not the excerpt.

:::{note}
The MACE checkpoint stays outside the image. `scripts/run-alchemi.sh`
selects it explicitly and mounts it read-only. This keeps the image and
model identities independent and avoids rebuilding the SIF for each model.
:::

The base image digest, both lock files, and the resulting SIF identity all
matter. A successful `%test` confirms that the expected packages import; a
short GPU MD run in a real allocation is still needed before using a new
image in the exercise.
