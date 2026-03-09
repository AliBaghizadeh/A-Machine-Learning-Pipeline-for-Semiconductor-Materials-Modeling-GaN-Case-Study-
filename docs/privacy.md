# Privacy / Offline-Only Policy

This repository is designed to run locally and offline.

## What this demo does
- Reads local DFT/MLIP artifacts produced on your machine.
- Runs a Streamlit UI locally in "demo mode" using precomputed artifacts from `app/demo_data/`.
- Optionally runs a narrow, local-only RAG sidecar using **Ollama on localhost** (if installed).

## What this demo does NOT do
- No external API calls.
- No telemetry.
- No cloud uploads.
- No remote vector databases.

## Data locations
- DFT results: `dft/results/`
- MLIP datasets: `mlip/data/datasets/`
- MLIP training runs: `mlip/results/`
- Demo artifacts: `analysis/artifacts/` and `app/demo_data/`
- Local papers for RAG: `rag/sources/`

