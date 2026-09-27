# Open a notebook from your computer

JupyterLab runs on the allocated GPU node. Your browser runs on your own
computer. An SSH tunnel connects a browser port to a loopback-only Jupyter
port; the browser must still provide Jupyter's private token. Do not bind
Jupyter to a public network address or place the token in a shared document.

Start with `scripts/start-jupyter.sh` inside the allocation. A connection
script must use the actual allocated node and the site's permitted SSH route.
On a site where direct SSH to that node is allowed, the shape is:

```bash
ssh -N -L 127.0.0.1:18888:127.0.0.1:8888 <allocated-node>
```

Then open the token URL in your browser, replacing its port with `18888`.
Some sites do not allow that direct route. In that case, use an approved
login-node relay with encryption and loopback binding, or use the static
handout. The instructor guide points to the session-specific connection
script; do not reuse an old job's node, certificate, or token.

Double-click a lesson `.md` file in JupyterLab. The provided launch script
sets MyST Markdown to open as a Jupytext notebook by default. The Markdown
file remains the editable source; the course does not maintain `.ipynb`
copies. Notebook cells can run in any order only where a page explicitly says
so. GPU exercises should run sequentially in one kernel.
