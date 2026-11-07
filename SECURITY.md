# Security policy

Report vulnerabilities privately via the repository's Security tab when private reporting is enabled. Otherwise contact the [repository owner](https://github.com/manikantbindass) through a private channel. Include the affected commit, impact, and reproduction steps using synthetic inputs.

Keep credentials, private images, medical identifiers and raw diagnostic dumps out of Git, public issues, pull requests, and CI logs. If a secret is exposed, revoke or rotate it and contact the maintainer privately. The project currently reviews reports against the latest development branch.

Real patient data must never be committed. Use generated synthetic fixtures for tests and reproduction. The research upload workflow requires prior de-identification review. See the [threat model and imaging review guidance](docs/architecture/THREAT_MODEL.md) and [quality gates](docs/development/QUALITY.md).

> Research and educational use only. Not a medical device and not intended for clinical diagnosis or treatment.
