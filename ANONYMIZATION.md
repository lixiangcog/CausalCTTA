# Anonymization notes

This release was prepared for anonymous review. The following categories were
removed or replaced:

- Git history and remote repository metadata
- Personal usernames and machine-specific absolute paths
- Author and affiliation metadata
- Datasets, model checkpoints, generated pseudo labels, logs, and scheduler output
- Python bytecode and cache directories

Generated pseudo-label manifests use paths relative to the configured dataset
root. Before distribution, run `python verify_anonymity.py` from the package
root; the command exits with a non-zero status if common identity or artifact
patterns are detected.
