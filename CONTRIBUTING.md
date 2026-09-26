# Contributing

Install with `python -m pip install -e '.[dev]'`, then run `python -m pytest`.
Keep the default SAE-only paper path separate from experimental extensions.
When changing numerical behavior, explain the formula and add a small test that
compares against an independent calculation. Update input documentation when
changing the public API or CLI.

Use synthetic inputs for bug reports and tests. Keep checkpoints, raw research
artifacts, machine-specific launch scripts, credentials, and evaluation predictions
out of commits. Report Python/PyTorch versions and the smallest command that
reproduces the problem. Build with `python -m build` before proposing packaging
changes.
