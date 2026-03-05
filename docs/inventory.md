# Repository Inventory (GaN DFT + MLIP)

This file is a quick, practical map of the current repository as it exists today.
It lists the main entrypoints (scripts), what they do, and the artifacts they read/write.

Scope note: this repo contains some legacy documentation text (e.g. Lu/Sc/Fe oxide wording)
that does not match the current GaN pivot. The pipeline code paths below are the current
sources of truth for GaN.

## Top-Level Entrypoints

### `run_pipeline.py`
- Purpose: Orchestrates the end-to-end demo pipeline: structure gen -> DFT -> dataset -> MACE training -> analysis/validation.
- Stages and stage scripts:
  - `structures`: `dft/scripts/structure_generation.py`
  - `dft_tier_a`: `dft/scripts/tier_a_relaxation.py`
  - `dft_tier_b`: `dft/scripts/tier_b_calculations.py`
  - `extract_data`: `dft/scripts/extract_dft_data.py`
  - `train_mlip`: `mlip/scripts/train_mlip.py`
  - `active_learning`: `mlip/scripts/active_learning.py`
  - `analysis`: `analysis/scripts/structural_analysis.py`
  - `validation`: `analysis/scripts/compare_dft_mlip.py`
- Key outputs (by stage): see sections below.

## DFT (GPAW) Layer

### `dft/config/gpaw_params.py`
- Purpose: Centralized GPAW parameters + helper utilities:
  - k-point scaling helpers
  - relaxation defaults
  - GPU env setup helper(s)

### `dft/config/dft_budget.py`
- Purpose: Budget tracking for Tier-A/Tier-B counts and caps.

### `dft/scripts/structure_generation.py`
- Purpose: Generate GaN bulk/defect supercells and write CIFs into `dft/structures/`.
- Outputs:
  - `dft/structures/*.cif`
  - (typically) `dft/structures/structure_info.json` (structure ids + tags/metadata)

### `dft/scripts/tier_a_relaxation.py`
- Purpose: Tier-A full relaxations (high cost, low count).
- Outputs:
  - `dft/results/tier_a_results.json`
  - `dft/results/logs/*`
  - `dft/results/trajectories/*`
  - `dft/results/checkpoints/*.gpw`

### `dft/scripts/tier_b_calculations.py`
- Purpose: Tier-B single-point and short-relax runs for labels (intentionally lightweight/iterative).
- Outputs:
  - `dft/results/tier_b_results.json` (cumulative)
  - `dft/results/tier_b_results__<run_tag>.json` (per-run snapshot)
  - `dft/results/logs/gpaw_tierb_sp__<run_tag>.out`
  - `dft/results/logs/gpaw_tierb_sr__<run_tag>.out`
  - `dft/results/trajectories/tierb_short_relax_<structure_id>.traj`
  - `dft/results/checkpoints/tierb_*_<structure_id>.gpw`

### `dft/scripts/dft_md_snapshots.py`
- Purpose: Short DFT-MD (GPAW) and export a few frames as CIFs (used to de-correlate bulk).
- Outputs:
  - `dft/results/trajectories/dft_md_<...>__<run_tag>.traj`
  - `dft/structures/<structure_id>__md300K_frameXXX__<run_tag>.cif`
  - `dft/structures/dft_md_latest.json` (pointer JSON for latest MD run + exported frames)

### `dft/scripts/mini_line_prototypes.py`
- Purpose: Build smaller “prototype” defect-line structures (used earlier to improve coverage).
- Outputs:
  - CIF(s) under `dft/structures/` (structure ids depend on script arguments)

## Dataset Extraction

### `dft/scripts/extract_dft_data.py`
- Purpose: Convert completed Tier-A/Tier-B results JSON into `extxyz` datasets for MACE training.
- Inputs:
  - `dft/results/tier_a_results.json`
  - `dft/results/tier_b_results.json`
- Outputs:
  - `mlip/data/datasets/dataset_<run_tag>/train.xyz`
  - `mlip/data/datasets/dataset_<run_tag>/val.xyz`
  - `mlip/data/datasets/dataset_<run_tag>/test.xyz`
  - `mlip/data/datasets/dataset_<run_tag>/all_data.xyz`
  - `mlip/data/datasets/dataset_<run_tag>/all_data_full.xyz`
  - `mlip/data/datasets/dataset_<run_tag>/dataset_stats.json`
  - `mlip/data/LATEST_DATASET.txt` (pointer to latest dataset directory)

## MLIP (MACE) Training

### `mlip/config/model_config.py`
- Purpose: Centralized MACE hyperparameters (model + training + GPU config).

### `mlip/scripts/train_mlip.py`
- Purpose: Train MACE on GPU using `mace_run_train` and the latest dataset pointer.
- Inputs:
  - `mlip/data/LATEST_DATASET.txt` (preferred)
  - `mlip/data/datasets/.../train.xyz`, `val.xyz`, `test.xyz`
- Outputs:
  - `mlip/results/mace_run_<run_tag>/checkpoints/*.pt` + exported `*.model`
  - `mlip/results/mace_run_<run_tag>/logs/*`
  - `mlip/results/mace_run_<run_tag>/profiler/*` (optional profiler probe)
  - `mlip/models/gan_mace_compiled.model` (compiled deployable)
  - `mlip/results/training_summary.json` (summary written by the wrapper)

### `mlip/scripts/active_learning.py`
- Purpose: Active learning loop (sampling + DFT labeling + retraining). Not required for demo mode.

## Validation / Gates / Analysis

### `analysis/scripts/energy_gate.py`
- Purpose: Compare MLIP energy vs DFT Tier-B single-point energy for fixed “gate” structures.
- Inputs:
  - `dft/results/tier_b_results.json` (Tier-B single_point energies)
  - one or more CIFs (explicitly passed via `--case`)
- Outputs:
  - Prints pass/fail + `|dE|/atom` per case to stdout (loggable)

### `analysis/scripts/force_gate.py`
- Purpose: Compare MLIP forces vs DFT forces (Tier-B SP or SR), optionally localized by a coordination-based selection.
- Inputs:
  - `dft/results/tier_b_results.json` (forces)
  - CIF (or DFT geometry from trajectory if `--use-dft-geometry`)
- Outputs:
  - Prints force stats (overall + selected region) + pass/fail to stdout (loggable)

### Other analysis scripts
- `analysis/scripts/compare_dft_mlip.py`: broad DFT-vs-MLIP comparison utilities.
- `analysis/scripts/structural_analysis.py`: structural summaries.
- `analysis/scripts/plot_scf_convergence.py`: postprocess GPAW logs to convergence plots.

### Common analysis artifacts in this repo
- `analysis/results/` includes:
  - exported CIFs (e.g. `analysis/results/final_structures/*.cif`)
  - MLIP relaxation trajectories (e.g. `analysis/results/large_scale_mlip/*.traj`)
  - validation JSON(s) and plots

## Experiments / Logs (Human Record)

### `experiments/2026-02-28_experimental_plan_and_command_log.md`
- Purpose: The main lab notebook: commands, outputs, gate results, and conclusions.
- This is the current “source-of-truth” human-readable record of Phase-1/Phase-4 work.

## Literature / Papers (Local-Only)

### `literature/`
- Purpose: PDF collection for local-only parameter extraction (future RAG sidecar).
- Example contents: GaN DFT papers and general MLIP methodology papers.

## STEM / abTEM (Optional Demo Add-On)

### `STEM imag simulation/`
- Contains abTEM notebooks:
  - `STEM imag simulation/abTEM_fast_scan.ipynb`
  - `STEM imag simulation/abTEM_4D_STEM_flexible_detectors.ipynb`

## Known Legacy References

- `README.md` and `QUICKSTART.md` currently contain legacy oxide wording and refer to scripts that do not exist (e.g. `dft/scripts/run_dft_pipeline.py`, `analysis/scripts/analyze_results.py`). Treat `run_pipeline.py` + the directories above as the real entrypoints.

