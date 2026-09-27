# MyST rendering in JupyterLab

The lesson's `.md` files are the maintained notebooks. Jupytext opens them
in JupyterLab. The included patch to `jupyterlab-myst` is a bounded renderer
for this course's MyST callouts and relative `{literalinclude}` excerpts;
it is not a general Sphinx compatibility layer.

The patch targets upstream commit
`4b4b321907455394a9ab8ceada77fc4d6940cbca`. Apply it to that exact
source revision in a private Jupyter environment, not a site-wide install:

```bash
git clone https://github.com/jupyter-book/jupyterlab-myst.git
cd jupyterlab-myst
git checkout 4b4b321907455394a9ab8ceada77fc4d6940cbca
git apply --check /path/to/mlip-md-lesson/jupyterlab-enccs/jupyterlab-enccs.patch
git apply /path/to/mlip-md-lesson/jupyterlab-enccs/jupyterlab-enccs.patch
python -m pip install build 'jupyter-builder>=1.2,<2'
# Make a private Corepack shim directory so child build processes find pnpm.
mkdir -p /path/outside/git/corepack-shims
corepack prepare pnpm@11.17.0 --activate
corepack enable --install-directory /path/outside/git/corepack-shims
export PATH=/path/outside/git/corepack-shims:$PATH
pnpm install --frozen-lockfile
pnpm run build:prod
python -m build --wheel
```

Build this wheel on a development machine, then install it into a private
Jupyter environment on the site. The wheel is architecture-independent; the
notebook Python packages still have to match the site's architecture.

The file resolver uses Jupyter's authenticated, same-origin `/files/` route
under the selected server root. It rejects absolute paths, cross-origin
downloads, non-UTF-8 content, and files over 200 kB. It currently supports
`:language:`, one contiguous `:lines:` range, `:start-at:`, `:end-at:`, and
`:end-before:`. It accepts `:linenos:` and `:emphasize-lines:` for the
Sphinx source but presents plain code in Lab; highlighting and source line
numbers belong to the published page. Unsupported options leave an explicit
unavailable marker.
Keep the Jupyter server root at the lesson repository so included files are
inside the authorized file boundary.

The published Sphinx rendering remains the complete handout. A clean
JupyterLab visual and accessibility check of all pages is required before
using a patched extension in a class. Do not assume a local wheel build is
equivalent to a tested site installation.
