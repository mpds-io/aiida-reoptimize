### Requirement: Contributing guide
The project SHALL include a CONTRIBUTING.md file at the project root that documents how to set up the development environment, code style conventions, and the pull request process.

#### Scenario: New contributor sets up development environment
- **WHEN** a new contributor reads CONTRIBUTING.md
- **THEN** they can clone the repository, install development dependencies, and run the linter/formatter

#### Scenario: Contributor submits a pull request
- **WHEN** a contributor reads the PR process section of CONTRIBUTING.md
- **THEN** they understand the branching convention, commit message style, and review expectations

### Requirement: Code style documentation
CONTRIBUTING.md SHALL document the project's code style rules, including the ruff configuration (line-length, target Python version) and required checks before pushing.

#### Scenario: Contributor checks code style
- **WHEN** a contributor follows the code style section
- **THEN** they can run `ruff check` and `ruff format` to verify their changes comply with project conventions
