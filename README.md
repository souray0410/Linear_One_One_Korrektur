# Linear One-One Korrektur (LOOK)

This repository provides a dataset-independent numerical core for weighted feature representations and affine correction. It contains code and synthetic tests only. It does not contain datasets, checkpoints, experiment results, or deployment configurations.

## Scope

Available modules include sample-group weighting, feature normalization, weighted PCA, grouped ridge fitting, and paired stochastic moments. These are numerical components, not a complete training or model-selection pipeline. Research-specific host integrations and execution systems are maintained separately. The project name does not imply that every fitted matrix is invertible or one-to-one.

The current normalization first standardizes individual coordinates and then normalizes total node energy. PCA dimensions and ridge candidates must be chosen under an explicit training-only protocol. Grouped fitting alone does not prevent leakage from upstream feature extraction.

## Install and test

```sh
python -m pip install -e '.[test]'
python -m pytest
python examples/synthetic_arrays.py
```

Python imports use `look`; the repository name and installable distribution name are different. This numerical release depends only on NumPy and does not require a separate graph framework or access to private repositories.

## Data interface

`look.data.interface.AlignedFeatures` accepts named feature arrays, ordered opaque sample keys, group keys and optional missingness masks. All modalities must have exactly the same sample order. It checks finite values, row alignment and group consistency. It does not download data or automatically serialize sample metadata.

Implement source-specific loading outside this repository. Return arrays through the interface; keep original identifier mappings, permissions, labels, split manifests and generated predictions in authorized storage. Use training data for fitting and keep evaluation data separate. The example creates all arrays using a fixed random generator.

## Publication boundary

Only reviewed source files, synthetic examples and generic documentation belong here. Dataset names, cohort-specific adapters, individual records, derived participant-level outputs, operational logs and unpublished results must not be committed or attached to issues. Private hosting does not authorize restricted data sharing.

Run `python tools/check_public_release.py --history` before pushing. Install the local hook with `git config core.hooksPath .githooks`. The SHA256 allowlist is an integrity boundary for reviewed files, not proof that arbitrary edits are safe; additions require manual review and an updated manifest. This hook does not control other clones or direct API uploads. No hosted workflow is included.
