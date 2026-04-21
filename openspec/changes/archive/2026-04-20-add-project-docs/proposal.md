## Why

The project has a solid README but lacks structured, comprehensive documentation. There is no API reference, no dedicated usage guides for key workflows, no contributor guide, and the examples directory has no index or walkthrough. This makes it difficult for new users to onboard and for contributors to understand the codebase.

## What Changes

- Add a Sphinx-based documentation site with API reference auto-generated from docstrings
- Add structured usage guides covering dynamic workflows (`OptimizerBuilder`), static workflows, and structure utilities
- Add an examples index/README that catalogs and explains each example script
- Add a contributor guide (CONTRIBUTING.md) with setup instructions, code style, and PR process
- Add a CHANGELOG.md to track releases

## Capabilities

### New Capabilities

- `docs-site`: Sphinx documentation site with API reference, usage guides, and examples index
- `contributing-guide`: CONTRIBUTING.md with development setup, code style, and PR conventions
- `changelog`: CHANGELOG.md for tracking project releases and changes

### Modified Capabilities

## Impact

- New `docs/` directory at project root with Sphinx configuration and documentation sources
- New `CONTRIBUTING.md` and `CHANGELOG.md` at project root
- Existing docstrings in `aiida_reoptimize/` source files may need improvement for clean API reference generation
- No impact on runtime code or APIs
