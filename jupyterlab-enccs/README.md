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

The published Sphinx rendering remains the complete handout. The patch is
complete for the directives this lesson uses, not a general Sphinx renderer.
The JupyterLab DOM uses `.jp-MarkdownCell .myst`; styling that targets an
imagined `.jp-RenderedMySTMarkdown` wrapper does not apply. The lesson patch
uses the observed selector and scopes it to notebook content.

For a private JupyterLab test session with the patched wheel installed,
`check_rendering.py` checks every episode, setup and reference page plus a
fixture containing all three ENCCS callouts and a relative literal include:

```bash
python -m pip install selenium
python jupyterlab-enccs/check_rendering.py \
  --base-url http://127.0.0.1:8877 \
  --token-file /path/to/private/token \
  --geckodriver /path/to/geckodriver
```

The September 2026 ASUS browser check passed all 16 current pages, including
the JUPITER setup page, in an isolated JupyterLab workspace. The Arrhenius
private installation reported both `jupyterlab-myst` and `jupyterlab-jupytext`
healthy. The upstream source's two unit suites passed (7 tests). This does
not establish general Sphinx-directive compatibility or screen-reader
acceptance; an instructor should still inspect the live notebook on their
target browser before class.
