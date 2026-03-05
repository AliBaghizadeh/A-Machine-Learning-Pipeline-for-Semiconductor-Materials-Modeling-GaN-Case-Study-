# GaN DFT + MACE MLIP (Offline “Production-Style” Demo)

This repository is an **offline-first** demo showcasing how to structure a local DFT → dataset → MLIP → validation loop in a production-style way.

Scope: This is a **demo**. Publication-grade physics is out-of-scope. The goal is to demonstrate an approach that can be scaled.

## What You Get
- A reproducible “golden run” artifact bundle (`analysis/artifacts/golden_run_<date>/run_manifest.json`)
- Model Card + Dataset Card generated from the manifest
- A Streamlit demo UI that runs fully offline using `app/demo_data/`
- A narrow, local-only RAG sidecar that extracts **typical GaN DFT parameters** from local text sources (PDF parsing is best-effort) and compares them to local configs + run manifest

## Local-Only / Privacy
No external API calls. Optional Ollama integration uses localhost only.
See `docs/privacy.md`.

## 3-Minute Demo
```bash
conda env create -f environment.yml
conda activate mlip_env

make demo-artifacts
make demo
```

Optional RAG:
```bash
make rag-ingest
make rag-report
```

## Repo Structure (Showcase)
- `docs/`: architecture + walkthrough + privacy + model/dataset cards
- `configs/`: YAML configs (DFT Tier-B, MACE training, gates)
- `scripts/`: demo utilities (freeze artifacts, generate cards)
- `app/`: Streamlit UI (offline demo mode)
- `rag/`: local-only RAG sidecar (papers → parameter table → report)
- `dft/`, `mlip/`, `analysis/`: existing pipeline code + outputs
- `experiments/`: human-readable command logs and outcomes

## Pipeline Entrypoint (Existing)
The current orchestration entrypoint is:
```bash
python run_pipeline.py --dry-run
```

For a concrete map of scripts and artifacts, see `docs/inventory.md`.
