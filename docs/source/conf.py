project = "aiida-reoptimize"
copyright = "2025-2026, Materials Platform for Data Science OU"
author = "Anton Domnin"

release = "0.8.10"
version = "0.8"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_mock_imports = [
    "aiida",
    "aiida.engine",
    "aiida.orm",
    "aiida.plugins",
    "aiida.common",
    "aiida_fleur",
    "aiida_crystal_dft",
    "pymoo",
    "mpds_client",
    "ase",
    "ase_fleur",
    "spglib",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "aiida": ("https://aiida-core.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
