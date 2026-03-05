# Architecture (Demo + Pipeline)

This is a demo-first, offline pipeline with a narrow RAG sidecar.

```mermaid
flowchart LR
  A[CIF / Structures] --> B[DFT Tier-B (GPAW)]
  B --> C[tier_b_results.json + traj + gpw]
  C --> D[extract_dft_data.py -> extxyz]
  D --> E[MACE training -> .model]
  E --> F[MLIP relax (optional, local)]
  E --> G[Validation gates]
  C --> G
  F --> G

  subgraph RAG Sidecar (Offline)
    P[Local papers: PDF/txt] --> Q[Ingest + chunk]
    Q --> R[Index + embeddings (Ollama optional)]
    R --> S[GaN DFT params table + citations]
    S --> T[Compare to local configs + run manifest]
  end
```

## Key Design Points
- Demo UI loads from `app/demo_data/` by default (no compute required).
- DFT is never run from the Streamlit app.
- RAG is intentionally narrow: it extracts typical GaN DFT parameters and compares them to local run configs/results.

