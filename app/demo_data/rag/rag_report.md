# RAG Report: GaN DFT Parameters (Offline Demo)
Generated: `2026-03-04T14:39:05.540438`
## Scope
- Narrow extraction of typical GaN DFT parameters from local papers/snippets.
- Compare those typical values to this repo's local run manifest (gates + dataset).

## Extracted Parameter Table (with citations)
| param | value | citation |
|---|---|---|
| xc_functional | PBE | demo_paper_snippet.txt#chunk0 |
| xc_functional | GGA | demo_paper_snippet.txt#chunk0 |
| kpoints | 6x6x4 | demo_paper_snippet.txt#chunk0 |

## Local Run Summary (from run_manifest.json)
- Energy gate pass: `True`
- Force gate pass: `True`
- Frozen model path: `/mnt/c/Ali/microscopy datasets/MLIP/mlip/results/mace_run_20260304_112648/checkpoints/gan_mace_run-42.model`
- Frozen dataset dir: `/mnt/c/Ali/microscopy datasets/MLIP/mlip/data/datasets/dataset_20260304_112646`

## Local Config Snapshot (best-effort)
- configs/dft_tierb.yaml gpu_ecut_eV: `None`
- configs/dft_tierb.yaml convergence energy: `None`
- configs/dft_tierb.yaml convergence density: `None`
- configs/dft_tierb.yaml convergence eigenstates: `None`

## Comparison (Typical vs Local)
| parameter | extracted typical | local config | within typical? |
|---|---:|---:|---|
| cutoff_eV | (not found) | None | unknown |
| convergence.energy | (not found) | None | unknown |
| convergence.density | (not found) | None | unknown |
| convergence.eigenstates | (not found) | None | unknown |
