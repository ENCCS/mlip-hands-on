"""Published pages do not execute GPU workloads."""

project = "Molecular dynamics with MACE on GPUs"
author = "ENCCS contributors"
extensions = ["sphinx_lesson", "sphinx_evita", "myst_nb", "sphinx_design"]
master_doc = "index"
exclude_patterns = ["_build", "AGENTS.md", "README.md", "THIRD_PARTY.md",
                    "jupyterlab-enccs/README.md", "jupyterlab-enccs/renderer-fixture.md"]
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence"]
html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "light_logo": "ENCCS_logo_light.png",
    "dark_logo": "ENCCS_logo_dark.png",
}

# sphinx-evita
evita_eu_funding_badge = "co-funded"
