# Build the ALCHEMI image

The ALCHEMI runtime and Python dependencies are described by
[`alchemi-aarch64.def`](../../alchemi-aarch64.def). The definition is for
ARM/GH200; the MACE weights stay outside the image.

```{literalinclude} ../../alchemi-aarch64.def
:language: text
:linenos:
:lines: 1-20
:emphasize-lines: 1-2,5-8,13-16
```

The base image is pinned by digest. `%files` copies the two hash-locked
dependency lists; `%post` installs from those lists. Keep
[`locks/build-requirements.lock`](../../locks/build-requirements.lock) and
[`locks/requirements.lock`](../../locks/requirements.lock) with the
definition when building.

The build command is short:

```bash
apptainer build "$MLIP_ALCHEMI_SIF" alchemi-aarch64.def
```

Select a fresh output path
on storage with enough space. Building an image may require a prepared
build environment; instructors can supply a prebuilt image instead.

One exact existing ARM/GH200 image was hash-verified and ran the short
single, batched, and relaxation examples on Arrhenius and JUPITER. The
current definition file keeps the same locked packages but omits an old
bundled example; rebuilding it produces a different image identity and
requires its own test. JUPITER compute nodes cannot fetch the Docker base
image directly; build on a permitted networked ARM64 host and copy the SIF
to JUPITER. The native JUPITER environment is a separate fallback.
