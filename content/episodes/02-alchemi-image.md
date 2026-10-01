# Build the ALCHEMI image

ALCHEMI runs the MACE model and MD integrator in a Python environment.
An Apptainer SIF packages that environment for use on either GH200 site.
Its Python dependencies are described by
[`alchemi-aarch64.def`](../../alchemi-aarch64.def). The definition is for
ARM/GH200.

```{note}
Keep MACE weights outside the SIF. The run command mounts the chosen model
read-only, so you can change models without rebuilding the environment.
```

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

JUPITER compute nodes cannot fetch the Docker base image directly. Build
on a permitted networked ARM64 host, then copy the SIF to project storage
on JUPITER. Before a session, run a short GPU example with the image you
will use. [Methods and checks](../reference/benchmarks.md) distinguishes
tested existing images from new builds.
