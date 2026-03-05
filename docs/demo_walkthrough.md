# 3-Minute Offline Demo Walkthrough

## Project Goal

This repo is a demo of a minimal closed-loop materials AI workflow:

`CIF/structures → DFT labeling → MLIP training → large-cell relaxation → validation gates → iterate`

It focuses on offline reproducibility: frozen artifacts (`run_manifest.json`), simple gates, and a narrow RAG sidecar.
See `docs/architecture.md` for the full diagram.

For the full step-by-step guide, see `docs/step_by_step_demo_guide.md`.

## Quick Demo
```bash
conda env create -f environment.yml
conda activate mlip_env
make demo-artifacts
make demo
```

## 1) Setup (once)
```bash
make setup
```

## 2) Build demo artifacts from your local runs (fast)
This collects a small subset of the latest artifacts and writes a `run_manifest.json`.
```bash
make demo-artifacts
```

## 3) Run the Streamlit demo (offline)
```bash
make demo
```

Gate meaning (very brief):
- Energy gate: `|ΔE|/atom <= 0.01 eV/atom` (DFT Tier‑B single‑point vs MLIP)
- Force gate: defect‑region `MAE <= 0.25 eV/A` and `MAX <= 1.0 eV/A` (DFT Tier‑B SR forces vs MLIP on SR geometry)

## 4) Optional: RAG (offline, local-only)
Put PDFs or `.txt` files into `literature/` (default source used by `make rag-ingest`).
If using PDFs, install parser support first: `pip install pypdf`.

Build index:
```bash
make rag-ingest
```

Generate report comparing literature parameters to your local run:
```bash
make rag-report
```

RAG outputs:
- `rag/index/gan_dft_params_table.csv` (structured params + citations)
- `analysis/artifacts/golden_run_<YYYYMMDD>/rag/rag_report.md` (comparison report)

Interactive showcase in app:
- Open `RAG Assistant`
- Set source directory to `literature/`
- Set model to `qwen3:4b`
- Click `Ingest + Build Index`, then `Ask RAG`

## 5) Optional: STEM (precomputed or generated)
STEM is included to show how predicted structures can connect to experimental observables (e.g. HAADF‑STEM via abTEM).
If you have precomputed images, place them in `app/demo_data/stem/` and open the STEM page in the app.
You can also generate one local HAADF-like fallback image directly from the STEM page.

If `abtem` fails in `mlip_env`, use a separate env:
```bash
conda create -n stem_env python=3.11 -y
conda activate stem_env
pip install abtem matplotlib ase numpy
```

Small sweep (recommended):
```bash
python analysis/scripts/stem_abtem_sweep.py --structure dft/structures/GaN_bulk_sc_2x2x2.cif --out-dir analysis/results/stem_sweep --defocus-list="-10,0,10,30" --inner-list 18,24 --outer-list 30,36 --semiangle-cutoff 36 --scan-divisor 4.0 --max-cases 8
```

High-quality bigger image (GPU, slower):
```bash
python analysis/scripts/stem_abtem_haadf.py --structure dft/structures/GaN_bulk_sc_2x2x2.cif --out-dir app/demo_data/stem --repeat 1,1,4 --energy-kv 300 --semiangle-cutoff 80 --cs-mm 1.3 --defocus-a 0 --haadf-inner 50 --haadf-outer 75 --potential-sampling 0.025 --scan-divisor 4.5 --device gpu --tag haadf_better_01
```
