# Quick Start (GaN Offline Demo Showcase)

This quickstart is for the **offline demo UI + RAG sidecar**. It does not run DFT.

## Prerequisites
- Conda
- (Optional) GPU for faster MLIP evaluation
- (Optional) Ollama running on localhost for embeddings

## 1) Create the environment
```bash
conda env create -f environment.yml
conda activate mlip_env
```

## 2) Freeze demo artifacts from your local runs (fast)
```bash
make demo-artifacts
```

This writes:
- `analysis/artifacts/golden_run_<YYYYMMDD>/run_manifest.json`
- Refreshes `app/demo_data/run_manifest.json`
- Updates `docs/model_card.md` and `docs/dataset_card.md`

## 3) Run the Streamlit demo (offline)
```bash
make demo
```

## 4) Optional: RAG (offline, local-only)
Put `.txt` files (or PDFs if you install a PDF parser) into `rag/sources/`.
If `rag/sources/` is empty, the demo uses `app/demo_data/rag/demo_paper_snippet.txt`.

```bash
make rag-ingest
make rag-report
```

## 5) Existing pipeline entrypoint (optional, not part of demo)
```bash
python run_pipeline.py --dry-run
```

See `docs/inventory.md` for the full script/artifact map.
