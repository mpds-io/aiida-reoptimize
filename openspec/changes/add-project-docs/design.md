## Context

aiida-reoptimize is an AiiDA plugin that bridges AiiDA workchains with PyMOO optimization for crystal structure optimization. The project has a comprehensive README (236 lines) but no structured documentation site, no API reference, no contributor guide, and no changelog. The examples directory contains ~20 scripts across 8 subdirectories with no index or explanation.

The codebase uses Sphinx-compatible docstrings inconsistently. Some classes and functions have docstrings, but many lack them, and none are structured for clean autodoc generation.

## Goals / Non-Goals

**Goals:**
- Provide a Sphinx documentation site with API reference, usage guides, and examples index
- Create CONTRIBUTING.md with development setup, code style, and PR conventions
- Create CHANGELOG.md for release tracking
- Improve existing docstrings to support clean API reference generation
- Make the project accessible to new users and contributors

**Non-Goals:**
- Deploy the documentation site to a hosting platform (ReadTheDocs, GitHub Pages, etc.)
- Rewrite the existing README
- Add tests or test documentation (separate concern)
- Change any runtime code behavior

## Decisions

### D1: Sphinx as documentation framework

**Choice**: Sphinx with `sphinx-autodoc-typehints` and `sphinx-rtd-theme`.

**Rationale**: Sphinx is the standard for Python scientific/AiiDA ecosystem projects. autodoc-typehints handles the type annotations already present in the codebase. ReadTheDocs theme is familiar to the target audience.

**Alternatives considered**:
- MkDocs with mkdocstrings: Simpler config but less standard in AiiDA/scientific Python ecosystem.
- Jupyter Book: Good for tutorials but overkill for API reference.

### D2: Documentation structure

**Choice**: Organize docs as:
- `docs/source/index.rst` — landing page
- `docs/source/getting-started.rst` — installation and quick start
- `docs/source/dynamic-workflows.rst` — OptimizerBuilder usage guide
- `docs/source/static-workflows.rst` — pre-registered workchain guide
- `docs/source/optimizers.rst` — optimizer algorithm reference (parameters, convergence)
- `docs/source/structure-utils.rst` — structure manipulation utilities
- `docs/source/examples.rst` — examples catalog
- `docs/source/api/` — autodoc API reference modules

**Rationale**: Mirrors the two-workflow paradigm (dynamic vs static) that is the central concept. Groups optimizers and structure utilities as distinct reference sections.

### D3: Docstring standard

**Choice**: Google-style docstrings with type annotations in the signature (not in the docstring). Added via `sphinx-autodoc-typehints`.

**Rationale**: Google style is concise and readable. Type hints in signatures are already used throughout the codebase. `sphinx-autodoc-typehints` renders them cleanly without duplicating in docstring text.

### D4: Examples index approach

**Choice**: A single `docs/source/examples.rst` page with a table of all examples, grouped by category, with brief descriptions and links to source files.

**Rationale**: Easier to maintain than per-example pages. Users typically want to browse available examples and then look at the script directly.

## Risks / Trade-offs

- [Docstrings need improvement] → Many existing docstrings are minimal or missing. The initial pass will add basic docstrings; full detailed docs can be iteratively improved.
- [Sphinx build complexity] → Use minimal extensions to keep the build simple. Avoid custom directives or complex intersphinx setups initially.
- [Maintenance burden] → Documentation near code (docstrings) is easiest to keep current. Usage guides may drift; mitigate by keeping guides concise and linking to examples.
