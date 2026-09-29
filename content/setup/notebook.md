# Open the MyST notebook

The `.md` pages are the notebooks. Use a private JupyterLab environment
with Jupytext and the lesson's bounded MyST renderer. Start the server inside
an allocated GPU step and open its authenticated URL through your site's
approved SSH forwarding route. Do not share the token or expose a public
listener.

Export your private `.env` and source `scripts/${MLIP_SITE}-lammps-env.sh`
before starting JupyterLab, so the kernel inherits artifact paths and the
matching native toolchain. Open a lesson `.md` as a Jupytext notebook.
The runnable cells show the same short commands printed on the page.

Published HTML does not execute GPU cells. The instructor guide covers
private server and forwarding checks.
