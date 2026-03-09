# A Machine Learning Pipeline for Semiconductor Materials Modeling (GaN Case Study)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![GPAW](https://img.shields.io/badge/DFT-GPAW-0B7285?style=for-the-badge)
![MACE](https://img.shields.io/badge/MLIP-MACE-E67E22?style=for-the-badge)
![Ollama](https://img.shields.io/badge/LLM-Ollama-1F77B4?style=for-the-badge)

This repository presents a lightweight workflow for semiconductor materials modeling, using GaN as the main case study. It connects density functional theory (DFT) reference calculations with machine-learned interatomic potentials (MLIPs) so larger atomic systems can be explored faster without giving up basic validation against physics-based results.

The project is structured as a practical pipeline rather than a loose collection of scripts. It includes dataset extraction, ML potential training, large-cell relaxation, validation gates, and a local demo interface for reviewing results, making it suitable both for reproducible research and for technical demonstrations of an end-to-end materials AI workflow.

Only the demo interface is offline-first by default. The full project is compute-oriented and can be run on local workstations, HPC clusters, or cloud GPU environments depending on your DFT and ML training needs.

Reproducible codebase for a practical workflow:

`Crystal structure -> DFT reference labeling -> dataset extraction -> ML potential training -> large-cell relaxation -> quality checks`

This repository focuses on **code, scripts, and configuration**. Heavy results and local artifacts are intentionally excluded from version control.

![Project pipeline](app/demo_data/plots/Pipleline.png)

## What This Project Demonstrates
- A clean end-to-end workflow from physics-based reference calculations (DFT) to fast ML-driven structure exploration.
- A production-style project layout with reproducible CLI entry points.
- Offline Streamlit demo pages for pipeline status, quality checks, relaxation demo, and RAG summary.
- Practical guardrails: validation gates before using the ML model on larger systems.

## Why This Matters
- DFT is accurate but expensive for large cells.
- ML potentials are fast but must be checked against reference calculations.
- Combining both enables scalable, credible materials modeling for semiconductor defect studies.

## How This Scales and Extends
- **Larger structures:** train on curated DFT labels, then run ML relaxations on much larger supercells that are impractical for routine DFT-only loops.
- **Active learning:** add uncertain or failure-case structures back into DFT labeling, then retrain to improve model coverage iteratively.
- **Metrology workflows:** connect relaxed structures to synthetic observables (for example STEM-like simulations) to support interpretation of experimental signals.
- **Materials design workflows:** use the trained potential for fast screening of defect configurations, strain states, and composition variants before expensive DFT confirmation.
- **Deployment flexibility:** keep the same scripts/configs while scaling execution from laptop prototyping to cluster/cloud orchestration.

![DFT to MLIP workflow](app/demo_data/plots/DFT%20to%20MLIP.png)

## Core Packages
- [MACE (mace-torch)](https://github.com/ACEsuit/mace) - ML interatomic potential framework used for training/inference
- [PyTorch](https://pytorch.org/) - deep learning backend for MACE
- [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/) - structure I/O and atomistic optimization workflow
- [GPAW](https://gpaw.readthedocs.io/) - DFT reference calculations
- [Streamlit](https://streamlit.io/) - local demo UI
- [abTEM](https://abtem.readthedocs.io/) *(optional)* - STEM image simulation

## Related Article
For a higher-level overview of the project, see:

[From DFT to MLIP: A Lightweight Materials Modelling Pipeline for Semiconductor Metrology](https://medium.com/@alibaghizade/from-dft-to-mlip-a-lightweight-materials-modelling-pipeline-for-semiconductor-metrology-a6a6958674a3)

## Repository Structure
```text
.
|-- app/                     # Streamlit demo application
|   |-- main.py
|   |-- pages/
|   `-- demo_data/plots/     # static images used in README/app
|-- analysis/
|   `-- scripts/             # quality checks and analysis
|-- configs/                 # YAML configs for training/gates/DFT settings
|-- dft/
|   |-- scripts/             # DFT labeling / structure utilities
|   `-- structures/          # input CIF structures
|-- mlip/
|   `-- scripts/             # ML potential training/AL scripts
|-- rag/                     # local RAG ingest/report components
|-- scripts/                 # artifact freezing / cards generation
|-- tests/                   # smoke tests
|-- run_pipeline.py          # pipeline orchestration entry point
|-- Makefile
`-- environment.yml
```

## Quick Start
```bash
conda env create -f environment.yml
conda activate mlip_env
make demo
```

## Core Commands
```bash
# DFT labeling (reference data)
dft/scripts/tier_b_calculations.py --gpu --calc-type single_point --structure-ids GaN_bulk_sc_2x2x2 --max-structures 1 --maxiter 25 --conv-energy 1e-3 --conv-density 1e-2 --conv-eigenstates 1e-4 --mag-config none

# Build ML dataset from completed DFT results
dft/scripts/extract_dft_data.py --max-atoms 300

# Train ML potential
mlip/scripts/train_mlip.py --max-epochs 80 --patience 10 --eval-interval 20 --energy-weight 10.0 --forces-weight 10.0

# Gate checks
analysis/scripts/energy_gate.py --model path/to/model.model --dft-json dft/results/tier_b_results.json --device cuda --threshold 0.01 --case GaN_bulk_sc_4x4x4_relaxed:dft/structures/GaN_bulk_sc_4x4x4_relaxed.cif
analysis/scripts/force_gate.py --model path/to/model.model --structure-id GaN_vacancy_line_N_sc_4x4x4_relaxed --cif dft/structures/GaN_vacancy_line_N_sc_4x4x4_relaxed.cif --source sr --use-dft-geometry --select coord --coord-rcut 2.4 --coord-max 3 --mae-thresh 0.25 --max-thresh 1.0
```

## Reproducibility and Data Policy
- Computed outputs are not versioned (e.g., `dft/results/`, `mlip/results/`, `analysis/results/`).
- Keep this repository code-centric; regenerate artifacts on your own compute resources.
- Use `make demo-artifacts` when you want to refresh local demo bundles.

## Streamlit Demo
```bash
make demo
```
Open the shown local URL to explore pipeline status, quality checks, ML relaxation demo, and the RAG assistant.

## License
This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
