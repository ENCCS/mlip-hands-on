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

# Open a notebook from your computer

JupyterLab runs on the allocated GPU node. Your browser runs on your own
computer. SSH encrypts the laptop-to-login hop; the Arrhenius launch script
also enables TLS for the login-to-compute hop. Jupyter requires its private
token. Keep the token and the generated certificate key out of Git.

Start the bounded job with `scripts/submit-arrhenius-jupyter.sh`, as described
in [Arrhenius setup](arrhenius.md). On your laptop, the included script looks
up the node of that running Slurm job and forwards through the login host:

```bash
bash scripts/connect-from-laptop.sh notebook <login-ssh-alias> <job-id>
```

Then open the private token URL printed by Jupyter, replacing its port with
`18888` and keeping `https://`. The script never reads or stores the token.
The certificate is self-signed. Before forwarding, the connection script
compares the live compute-node certificate with the owner-private
`jupyter-<job-id>.fingerprint` beside the job log. It refuses a mismatch. The
browser may still warn because this short-lived certificate is not signed by
a public authority; confirm the script's successful check before accepting
that warning. An expired job must not be reused.

The live HTML handout is a separate loopback service. From your laptop, use:

```bash
bash scripts/connect-from-laptop.sh html <html-host-ssh-alias>
```

This maps the host's port `8766` to `http://127.0.0.1:18766/` on your laptop.
Change the local or remote ports with the optional arguments if necessary.

Double-click a lesson `.md` file in JupyterLab. The provided launch script
sets MyST Markdown to open as a Jupytext notebook by default. The Markdown
file remains the editable source; the course does not maintain `.ipynb`
copies. Notebook cells can run in any order only where a page explicitly says
so. GPU exercises should run sequentially in one kernel.
