"""Published pages do not execute GPU workloads."""

project = "Universal MLIPs on HPC: hands-on"
html_title = project
author = "ENCCS contributors"
extensions = ["sphinx.ext.githubpages", "sphinx.ext.intersphinx", "sphinx_lesson", "sphinx_evita",
              "myst_nb", "sphinx_design"]
master_doc = "index"
exclude_patterns = ["_build", "**/.pixi", "README.md", "THIRD_PARTY.md",
                    "jupyterlab-enccs/README.md", "jupyterlab-enccs/renderer-fixture.md"]
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence"]
html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_css_files = ["overrides.css", "funding.css"]
copyright = (
    "ENCCS contributors | Co-funded by the European Union (EuroCC 3, grant agreement No. 101306701). "
    "Views and opinions expressed are however those of the author(s) only and do not necessarily reflect "
    "those of the European Union or the granting authority (European High-Performance Computing Joint "
    "Undertaking: EuroHPC JU). Neither the European Union nor the granting authority can be held "
    "responsible for them."
)
html_theme_options = {
    "light_logo": "ENCCS_logo_light.png",
    "dark_logo": "ENCCS_logo_dark.png",
    "source_repository": "https://github.com/ENCCS/mlip-hands-on/",
    "source_branch": "main",
    "source_directory": "",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ENCCS/mlip-hands-on",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# sphinx-evita: ENCCS lessons use only the EU funding badge, not the EVITA branding
import logging

evita_eu_funding_badge = "co-funded"
logging.getLogger("sphinx.sphinx_evita").addFilter(
    lambda record: "not detected as an EVITA project" not in record.getMessage())

# Plausible analytics, as in the ENCCS lesson template; only for the published main build
import os

if os.environ.get("GITHUB_REF", "") == "refs/heads/main":
    html_js_files = [("https://plausible.io/js/script.js",
                      {"data-domain": "enccs.github.io/mlip-hands-on", "defer": "defer"})]
