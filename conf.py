"""Published pages do not execute GPU workloads."""

project = "Universal MLIPs on HPC: hands-on"
html_title = project
author = "ENCCS contributors"
extensions = ["sphinx_lesson", "sphinx_evita", "myst_nb", "sphinx_design"]
master_doc = "index"
exclude_patterns = ["_build", "**/.pixi", "README.md", "THIRD_PARTY.md",
                    "jupyterlab-enccs/README.md", "jupyterlab-enccs/renderer-fixture.md"]
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence"]
html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_css_files = ["funding.css"]
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
}

# sphinx-evita: ENCCS lessons use only the EU funding badge, not the EVITA branding
import logging

evita_eu_funding_badge = "co-funded"
logging.getLogger("sphinx.sphinx_evita").addFilter(
    lambda record: "not detected as an EVITA project" not in record.getMessage())
