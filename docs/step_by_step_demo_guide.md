# Step-by-Step Instruction Guide (Offline Demo)

## Project Goal (Read This First)

This project demonstrates a minimal **closed-loop materials AI pipeline**:

`CIF/structures → DFT labeling → MLIP training → large-cell relaxation → validation gates → iterate`

What the demo shows (offline, reproducible):
1. How a DFT→MLIP workflow can be packaged like a production project (clean structure, configs, tests).
2. How we track artifacts with a frozen `run_manifest.json` (“golden run”) plus Model/Dataset Cards.
3. How we apply simple validation gates (energy + force) to decide whether an MLIP is “good enough” for demo use.
4. A narrow RAG assistant that extracts **typical GaN DFT parameters** from local papers/snippets and compares them to our local configs/results.

Why MLIP matters (high level):
- DFT is accurate but expensive; MLIPs approximate DFT forces/energies cheaply, enabling **large supercell relaxations** and fast screening.

Architecture reference:
- See `docs/architecture.md` for the pipeline diagram.

## Quick Demo (Fast Path)

If you just want to see the app working:
```bash
conda env create -f environment.yml
conda activate mlip_env
make demo-artifacts
make demo
```

## Demo Mode Policy

This guide is written for **demo mode** (offline-first). It does **not** run DFT.

## 0) Start In The Repo Root

Run everything from the repository root directory.

If your path contains spaces (common on `/mnt/c/...`), always quote it:
```bash
cd "/mnt/c/Ali/microscopy datasets/MLIP"
```

## 1) Create / Activate The Conda Environment

Create the environment (first time only):
```bash
conda env create -f environment.yml
```

Activate it (every new terminal):
```bash
conda activate mlip_env
```

Quick sanity check:
```bash
python -c "import sys; print(sys.version)"
python -c "import ase; print('ase OK')"
python -c "import torch; print('torch OK, cuda=', torch.cuda.is_available())"
```

### If Streamlit Is Missing
If `make demo` fails with `ModuleNotFoundError: streamlit`, install it into the active env:
```bash
python -m pip install streamlit
```

## 2) Freeze “Golden Run” Demo Artifacts (Fast)

This step packages a minimal, reproducible subset of your latest local outputs into:
- `analysis/artifacts/golden_run_<YYYYMMDD>/run_manifest.json`
- and refreshes `app/demo_data/run_manifest.json` so the UI works offline

Run:
```bash
make demo-artifacts
```

What this does:
1. Detects the latest trained MACE `*.model` and latest dataset pointer.
2. Re-runs the validation gates (energy + force) quickly (CPU fallback if needed).
3. Copies a small subset of CIFs/plots/JSON into the golden-run folder.
4. Writes/updates Model Card + Dataset Card in `docs/`.

### What “DFT Tier-B” Means (In This Repo)

This repo uses a simple two-tier concept:
- **Tier-A**: expensive, high-fidelity DFT (full relaxations, low count).
- **Tier-B**: cheaper, higher-throughput DFT meant to generate *labels* for MLIP training and spot-check gates.

In practice, **Tier-B is the workhorse** for this demo because it is fast enough to run iteratively and still produces the key supervision signals:
- total energies
- atomic forces
- (optionally) stress

Tier-B has two calculation modes:
1. **Tier-B Single-Point (SP)**  
   - One SCF calculation on a fixed geometry (no ionic relaxation).  
   - Purpose: generate clean energy/forces labels cheaply for many structures.  
   - Used by:
     - training dataset extraction (`dft/scripts/extract_dft_data.py`)
     - **energy gate** reference energies (`analysis/scripts/energy_gate.py`)

2. **Tier-B Short Relax (SR)**  
   - A *bounded* ionic relaxation (few steps, loose-ish tolerances).  
   - Purpose: add off-equilibrium / slightly-relaxed configurations so the MLIP learns more realistic local distortions (especially near defects).  
   - Used by:
     - training dataset extraction (source=`tier_b_sr`)
     - **force gate** reference forces and geometry (`analysis/scripts/force_gate.py --source sr --use-dft-geometry`)

Why SR geometry matters:
- Forces are gradients at a specific geometry. If you compare DFT forces from SR to MLIP forces on a different CIF geometry, you can get a false failure.
- In this repo, the force gate is designed to use the **same SR final structure** that produced the stored DFT forces.

Where Tier-B parameters live (source-of-truth vs demo snapshot):
- Source-of-truth implementation: `dft/scripts/tier_b_calculations.py` + `dft/config/gpaw_params.py`
- Demo-friendly config snapshot (for RAG comparison/reporting): `configs/dft_tierb.yaml`

Key Tier-B artifacts you’ll see on disk:
- DFT results database:
  - `dft/results/tier_b_results.json` (cumulative)
  - `dft/results/tier_b_results__<run_tag>.json` (per-run snapshot)
- Logs:
  - `dft/results/logs/gpaw_tierb_sp__<run_tag>.out`
  - `dft/results/logs/gpaw_tierb_sr__<run_tag>.out`
- Relaxation trajectories (SR):
  - `dft/results/trajectories/tierb_short_relax_<structure_id>.traj`
- Checkpoints (for restart/debug):
  - `dft/results/checkpoints/tierb_*_<structure_id>.gpw`

### What The Validation Gates Mean (Brief)

Validation gates are **spot checks**: they compare MLIP predictions against stored DFT reference results for a few representative structures.

Energy gate:
- Meaning: compare MLIP energy to DFT Tier-B single-point energy.
- Pass criterion (demo): `|ΔE|/atom <= 0.01 eV/atom` for all configured cases.

Force gate:
- Meaning: compare MLIP forces to DFT forces (Tier-B **short relax** by default in this repo), using the **DFT SR final geometry**.
- Pass criterion (demo): selected‑region force `MAE <= 0.25 eV/A` and `MAX <= 1.0 eV/A`.

Why this matters:
- Passing the energy gate reduces the risk of global energy drift on representative snapshots.
- Passing the force gate increases confidence that MLIP relaxations are locally meaningful near defects.

## 3) Run The Streamlit Demo (Offline)

Start the UI:
```bash
make demo
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

Recommended page order:
1. **Home**: privacy + architecture + quick summary
2. **Pipeline Status**: model/dataset pointers and “exists?” checks
3. **Validation Gates**: energy/force gate results + raw stdout logs
4. **MLIP Relaxation (Demo)**: view saved examples (no compute) and optionally run a short local relaxation
5. **RAG Assistant**: view latest RAG report and demo snippet
6. **STEM**: shows precomputed images if present in `app/demo_data/stem/`

## 4) Run The RAG Sidecar (Offline, Narrow Scope)

### 4.1 Add paper sources (optional)
Place papers into:
- `literature/` (default used by `make rag-ingest`)

Best supported right now:
- `.txt` files
- `.pdf` files if `pypdf` is installed (`pip install pypdf`)

### 4.2 Ingest + build index
```bash
make rag-ingest
```

If `literature/` is empty or unparseable, it automatically falls back to the bundled demo snippet:
- `app/demo_data/rag/demo_paper_snippet.txt`

Outputs:
- `rag/index/chunks.jsonl`
- `rag/index/index.json`
- `rag/index/gan_dft_params_table.csv`

### 4.3 Generate the report (compare typical params to local config/results)
```bash
make rag-report
```

Outputs (golden-run preferred):
- `analysis/artifacts/golden_run_<YYYYMMDD>/rag/rag_report.md`
- `analysis/artifacts/golden_run_<YYYYMMDD>/rag/gan_dft_params_table.csv`

### 4.4 Interactive RAG in Streamlit (showcase mode)
Use the **RAG Assistant** page:
- `Paper source directory`: set to `literature/`
- Click `Ingest + Build Index`
- `Ollama model`: set `qwen3:4b`
- Click `Ask RAG` with your question

Why this is useful:
- `gan_dft_params_table.csv`: a **structured** table of extracted GaN DFT parameters with per‑row citations.
- `rag_report.md`: a short report that **compares** extracted “typical” ranges to this repo’s local config + run manifest.

## 5) Run Smoke Tests (Offline)

Run:
```bash
make test
```

This validates:
- demo artifacts present
- app import works without Streamlit (import-only)
- RAG schema/table creation works without Ollama

## 6) Optional: Add STEM Images To The Demo

Why STEM exists in this project:
- STEM images are an example of connecting **predicted atomic structures** to an **experimental observable**.
- In a full pipeline, we can simulate HAADF‑STEM (e.g. via abTEM) using MLIP‑relaxed structures.

If you already have precomputed STEM images, copy them to:
```text
app/demo_data/stem/
```

Supported formats: `png`, `jpg`, `jpeg`.

Then open the **STEM** page in Streamlit.
You can also click `Generate HAADF-like image` on that page for a local, lightweight fallback image.

### 6.1 Real STEM script (abTEM)
Primary CLI script:
- `analysis/scripts/stem_abtem_haadf.py`

Environment note:
- If your main `mlip_env` is Python 3.10, run STEM in a separate Python 3.11 env:
```bash
conda create -n stem_env python=3.11 -y
conda activate stem_env
pip install abtem matplotlib ase numpy
```

Why a separate `stem_env` is needed:
- The current MLIP/DFT workflow is pinned to Python 3.10 for compatibility with your existing stack.
- Recent `abtem` builds use typing features (`typing.Self`) that require Python >= 3.11.
- Mixing these in one environment often causes import/version conflicts.
- Keeping `mlip_env` and `stem_env` separate makes the demo reproducible and avoids breaking your DFT/MLIP pipeline.

Example command:
```bash
python analysis/scripts/stem_abtem_haadf.py --structure dft/structures/GaN_bulk_sc_2x2x2.cif --out-dir app/demo_data/stem --repeat 1,1,1 --energy-kv 300 --semiangle-cutoff 36 --cs-mm 1.3 --defocus-a 30 --haadf-inner 24 --haadf-outer 36 --potential-sampling 0.025 --scan-divisor 3.5 --tag haadf_demo_latest
```

Key user controls:
- `--structure`: input CIF/structure
- `--repeat`: supercell repeat `x,y,z`
- `--energy-kv`: accelerating voltage
- `--semiangle-cutoff`: probe cutoff angle (detectors must be <= this)
- `--cs-mm`: spherical aberration in mm
- `--defocus-a`: defocus in Angstrom
- `--haadf-inner` / `--haadf-outer`: detector annulus in mrad
- `--potential-sampling`: potential grid spacing (A)
- `--scan-divisor`: scan sampling = nyquist / divisor (larger => faster)

Outputs:
- `<tag>.png`, `<tag>.npy`, `<tag>.json` in `--out-dir`

### 6.2 Quick parameter sweep (recommended before app demo)
Sweep script:
- `analysis/scripts/stem_abtem_sweep.py`

Example:
```bash
conda activate stem_env
python analysis/scripts/stem_abtem_sweep.py --structure dft/structures/GaN_bulk_sc_2x2x2.cif --out-dir analysis/results/stem_sweep --defocus-list="-10,0,10,30" --inner-list 18,24 --outer-list 30,36 --semiangle-cutoff 36 --scan-divisor 4.0 --max-cases 8
```

Sweep outputs:
- `analysis/results/stem_sweep/sweep_results.csv` (ranked metrics)
- `analysis/results/stem_sweep/sweep_summary.json` (best case)

How to use sweep results:
1. Open `analysis/results/stem_sweep/sweep_results.csv`.
2. Keep only rows with `status=ok`.
3. Sort by `score` (descending) and inspect the top 2-3 PNGs.
4. Pick the best visual/score tradeoff.
5. Re-run `stem_abtem_haadf.py` once with the chosen parameters and output to `app/demo_data/stem/`.
6. Use those same values as defaults in the Streamlit STEM page for your showcase.

### 6.3 Large-Cell Fast Showcase Preset (GPU)
Use this when you want a large-cell image that still runs quickly enough for a live demo.

Reference command (confirmed):
```bash
python analysis/scripts/stem_abtem_haadf.py --structure dft/structures/GaN_bulk_sc_2x2x2.cif --out-dir app/demo_data/stem --repeat 3,3,3 --scan-start 0.0,0.0 --scan-end 0.35,0.35 --scan-divisor 16 --potential-sampling 0.10 --energy-kv 300 --semiangle-cutoff 80 --haadf-inner 50 --haadf-outer 75 --defocus-a 0 --device gpu --cmap gray --tag haadf_xxl_fast_333
```

Why this setting is a good demo tradeoff:
- `--repeat 3,3,3`: larger effective sample/cell.
- `--scan-end 0.35,0.35`: smaller field-of-view to keep runtime low.
- `--scan-divisor 16` + `--potential-sampling 0.10`: faster sampling settings.
- `--cmap gray`: grayscale rendering for a microscope-like look.

Expected runtime:
- Usually fast on GPU (seconds to low minutes, hardware-dependent).

Post-run verification:
```bash
python -c "import json; d=json.load(open('app/demo_data/stem/haadf_xxl_fast_333.json')); print('device=', d['device'], 'shape=', d['array_shape'])"
```

App note:
- The STEM page now prioritizes `haadf_xxl_fast_333` for the “Bulk GaN (relaxed)” scenario when present in `app/demo_data/stem/`.

## 7) Where To Look For Outputs

- Golden run bundle: `analysis/artifacts/golden_run_<YYYYMMDD>/`
- Demo UI data source: `app/demo_data/`
- Cards:
  - `docs/model_card.md`
  - `docs/dataset_card.md`

## 8) Troubleshooting

- `pytest: command not found`:
  - Use `make test` (it runs `python -m pytest`) and ensure you are in `mlip_env`.
- Gates show `pass: null`:
  - You likely ran `make demo-artifacts` outside the env that has `mace-torch` installed.
  - Fix: `conda activate mlip_env` then re-run `make demo-artifacts`.
- RAG shows no extracted rows:
  - Install PDF parsing support: `pip install pypdf`
  - Keep files in `literature/` and rerun `make rag-ingest`.
