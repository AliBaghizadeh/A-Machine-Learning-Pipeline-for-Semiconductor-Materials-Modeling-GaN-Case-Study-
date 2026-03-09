# Model Card: GaN MACE MLIP (Demo)

Generated: `2026-03-04T15:07:58.917689`

## Overview
This model card describes the *demo* MACE MLIP artifacts used in this repository.
It is intended to showcase an offline, production-style workflow (not publication-grade physics).

## Frozen Artifacts
- Frozen manifest: `/mnt/c/Ali/microscopy datasets/MLIP/analysis/artifacts/golden_run_20260304/run_manifest.json`
- Model path: `/mnt/c/Ali/microscopy datasets/MLIP/mlip/results/mace_run_20260304_112648/checkpoints/gan_mace_run-42.model`
- Dataset dir: `/mnt/c/Ali/microscopy datasets/MLIP/mlip/data/datasets/dataset_20260304_112646`
- DFT reference JSON: `/mnt/c/Ali/microscopy datasets/MLIP/dft/results/tier_b_results.json`

## Intended Use
- Offline demo of an MLIP pipeline: dataset extraction -> MACE training -> validation gates -> iteration.
- Optional local MLIP relaxations for quick geometry exploration (no DFT from the app).

## Metrics (Gates)
- Energy gate pass: `True` (threshold `0.01` eV/atom)
- Energy gate cases: `2`
- Force gate pass: `True`
- Force gate selection: `coord` (selected `37` / total `252`)

## Limitations
- Demo-only dataset sizes are small and correlated.
- Gate pass/fail is a narrow correctness check, not a universal accuracy guarantee.
- Results depend on local GPAW settings and the specific Tier-B reference calculations stored in the repo.

## Reproducibility
- Run `make demo-artifacts` to regenerate the manifest + cards from current local outputs.
