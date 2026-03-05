# Dataset Card: GaN DFT Labels for MLIP (Demo)

Generated: `2026-03-04T15:07:58.917689`

## Overview
This dataset card describes the *demo* DFT-derived labels used to train the MACE model.

## Frozen Artifacts
- Frozen manifest: `/mnt/c/Ali/microscopy datasets/MLIP/analysis/artifacts/golden_run_20260304/run_manifest.json`
- Dataset dir: `/mnt/c/Ali/microscopy datasets/MLIP/mlip/data/datasets/dataset_20260304_112646`

## Composition / Size (from dataset_stats)
- total_structures: `29`
- total_structures_full: `31`
- max_atoms_filter: `300`
- train/val/test: `23` / `2` / `4`
- sources: `{'tier_a': 0, 'tier_b_sp': 23, 'tier_b_sr': 6}`

## Label Types
- Energies (eV)
- Forces (eV/Angstrom)
- Stress (when available)

## Known Limitations
- Small, demo-scale dataset; not meant for general-purpose deployment.
- Coverage is biased toward the structures used in the gate/iteration loop.

## Reproducibility
- Dataset is produced by `dft/scripts/extract_dft_data.py` and tracked by `mlip/data/LATEST_DATASET.txt`.
