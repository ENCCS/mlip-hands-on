# MLIP lesson repository

This repository is the source of the published MyST lesson and its runnable
examples. Write participant-facing pages as practical how-to guides: show the
command or code, the expected result, and only the explanation needed to use
or interpret it. Keep scripts small. Put facilitation advice in the instructor
guide, not the episodes.
Do not depend on Remote Agent, Agent Workbench, or a separate hands-on
framework for the primary exercises.

The Markdown files are the notebook source. Do not maintain `.ipynb` copies.
Keep model weights, SIFs, native binaries, credentials, personal paths,
scheduler output, and live notebook tokens outside Git. Use placeholders in
public site examples and review rendered HTML for leaks.

A local build or static review does not authorize a GPU allocation or Slurm
submission. Before a site run, check the exact artifacts, account, partition,
reservation, time, and output target. Do not retry an uncertain submission.
Report functional results separately from benchmark and scientific evidence.

Run `bash -n` for shell files, `python -m py_compile` for examples, a
warning-as-error Sphinx build, and a privacy scan before committing. GPU
notebooks and scripts need a distinct authorized site test.
