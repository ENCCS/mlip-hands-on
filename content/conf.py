"""ENCCS Sphinx lesson configuration; published pages never execute GPU work."""

project = "Molecular dynamics with MACE on GPUs"
author = "ENCCS contributors"
copyright = f"2026, ENCCS, {author}"
github_user = "ENCCS"
github_repo_name = "mlip-hands-on"
github_version = "lesson/minimal-md-cli"
conf_py_path = "/content/"
extensions = [
    "sphinx.ext.githubpages",
    "sphinx_lesson",
    "sphinx_evita",
    "myst_nb",
    "sphinx_design",
]
master_doc = "index"
exclude_patterns = ["_build", "AGENTS.md", "README.md", "THIRD_PARTY.md",
                    "jupyterlab-enccs/README.md", "jupyterlab-enccs/renderer-fixture.md"]
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "attrs_inline", "substitution"]
myst_substitutions = {"author": author}
html_title = project
html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_css_files = ["overrides.css"]
html_theme_options = {
    "light_logo": "ENCCS_logo_light.png",
    "dark_logo": "ENCCS_logo_dark.png",
    "source_repository": f"https://github.com/{github_user}/{github_repo_name}",
    "source_branch": github_version,
    "source_directory": conf_py_path,
}
html_context = {
    "display_github": True,
    "github_user": github_user,
    "github_repo": github_repo_name,
    "github_version": github_version,
    "conf_py_path": conf_py_path,
}
evita_eu_funding_badge = "co-funded"
