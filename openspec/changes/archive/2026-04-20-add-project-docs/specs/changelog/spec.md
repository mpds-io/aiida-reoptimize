## ADDED Requirements

### Requirement: Changelog file
The project SHALL include a CHANGELOG.md file at the project root that tracks releases and notable changes, organized in reverse chronological order under version headers.

#### Scenario: User checks what changed in a release
- **WHEN** a user reads CHANGELOG.md
- **THEN** they see changes grouped by version with sections for Added, Changed, Fixed, and Removed

### Requirement: Changelog format
CHANGELOG.md SHALL follow the "Keep a Changelog" format with version headers using the project version from `pyproject.toml` and date stamps for released versions.

#### Scenario: New release entry
- **WHEN** a new version is released
- **THEN** a new version header with date is added to CHANGELOG.md following the existing format
