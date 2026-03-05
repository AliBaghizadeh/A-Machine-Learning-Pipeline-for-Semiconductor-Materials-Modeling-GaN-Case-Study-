# Project Plan: GPU-Accelerated DFT + MLIP for Lu₁₋ₓScₓFeO₃

## Executive Summary

This project implements a budget-limited DFT + active learning MLIP workflow for hexagonal Lu₁₋ₓScₓFeO₃ (x = 0.2, 0.3, 0.4, 0.5) using:
- **DFT**: GPAW with ASE (GPU-accelerated where beneficial)
- **MLIP**: MACE (PyTorch-based equivariant model)
- **MD Engine**: ASE-MD and LAMMPS

**Key Constraint**: Maximum 100-200 DFT calculations

---

## Phase 1: Environment Setup (Days 1-2)

### Tasks
- [ ] Create conda environment from `environment.yml`
- [ ] Install GPAW and download PAW datasets
- [ ] Configure GPU support for GPAW
- [ ] Install MACE and verify PyTorch CUDA support
- [ ] Test GPU detection with small calculation

### Success Criteria
- `conda activate mlip_env` works
- `gpaw test` passes
- `python -c "import torch; print(torch.cuda.is_available())"` returns True
- GPU memory is detected (16 GB)

### Artifacts
- `environment.yml` ✓
- Installation log in `docs/installation_log.txt`

---

## Phase 2: Structure Generation (Days 2-3)

### Tasks
- [ ] Read and validate CIF files (h-LuFeO3.cif, LSFO.cif)
- [ ] Generate supercells for x = 0.2, 0.3, 0.4, 0.5
- [ ] Create Sc substitution patterns (5-10 configs per composition)
- [ ] Apply small distortions for MLIP training diversity
- [ ] Validate structures with spglib

### Supercell Strategy
| x (Sc fraction) | Supercell | Lu atoms | Sc atoms | Configs |
|-----------------|-----------|----------|----------|---------|
| 0.0 | 2×2×1 | 24 | 0 | 1 (reference) |
| 0.2 | 5×5×1 | 150 | 30 | 5 |
| 0.3 | 10×3×1 | 180 | 54 | 5 |
| 0.4 | 5×5×1 | 150 | 60 | 5 |
| 0.5 | 2×2×1 | 24 | 12 | 5 |

### Success Criteria
- All structures generated and validated
- Space group preserved after substitution
- Files saved in `dft/structures/`

### Artifacts
- `dft/structures/` directory with all generated structures
- `dft/structures/structure_report.json`

---

## Phase 3: DFT Calculations (Days 3-10)

### Tier A: Full Relaxations (5 structures)
- Undoped LuFeO₃ (reference)
- x = 0.5 structure (LSFO.cif reference)
- 2 additional representative structures

**GPAW Parameters**:
```python
gpaw_params = {
    'mode': 'lcao',
    'basis': 'dzp',
    'xc': 'PBE',
    'kpts': (4, 4, 2),
    'h': 0.18,
    'spinpol': True,
    'occupations': 'smearing',
    'sigma': 0.05,
    'convergence': {'energy': 1e-6, 'forces': 0.05},
    'txt': 'gpaw.log'
}
# PBE+U: U_eff = 5.0 eV for Fe 3d
```

### Tier B: Short Relaxations/Single-Points (145 structures)
- Fixed-cell or short ionic relaxations
- Focus on force/energy sampling for MLIP

### DFT Budget Allocation
| Category | Count | Purpose |
|----------|-------|---------|
| Tier A full relax | 5 | Reference structures |
| Tier B single-point | 100 | Initial training set |
| Tier B short relax | 40 | Off-equilibrium data |
| Reserve for AL | 55 | Active learning loop |
| **Total** | **200** | Maximum budget |

### Success Criteria
- All Tier A calculations complete
- Tier B calculations cover all compositions
- Dataset ready for MLIP training

### Artifacts
- `dft/results/` with .traj and .json files
- `dft/results/dft_dataset.json` consolidated dataset

---

## Phase 4: MLIP Dataset Preparation (Day 10)

### Tasks
- [ ] Extract energies, forces, stresses from GPAW outputs
- [ ] Convert to MACE-compatible format
- [ ] Split into training/validation/test sets (80/10/10)
- [ ] Create metadata file with composition labels

### Dataset Statistics
- Training: 140 structures
- Validation: 18 structures
- Test: 18 structures
- Reserve for AL: 24 structures

### Artifacts
- `mlip/data/train.xyz`
- `mlip/data/val.xyz`
- `mlip/data/test.xyz`
- `mlip/data/dataset_stats.json`

---

## Phase 5: MLIP Training (Days 10-14)

### Model: MACE
- Equivariant neural network
- R_max = 5.0 Å
- 3 interaction layers
- 128 hidden channels
- L_max = 2 (max angular momentum)

### Training Parameters
```python
training_config = {
    'batch_size': 4,  # Limited by 16GB VRAM
    'learning_rate': 1e-4,
    'epochs': 500,
    'energy_weight': 1.0,
    'forces_weight': 10.0,
    'stress_weight': 0.1,
    'ema_decay': 0.99,
    'patience': 50,
}
```

### Success Criteria
- Energy MAE < 5 meV/atom
- Forces MAE < 50 meV/Å
- Validation loss stable

### Artifacts
- `mlip/models/mace_model.pt`
- `mlip/results/training_log.json`
- `mlip/results/training_curves.png`

---

## Phase 6: Active Learning Loop (Days 14-18)

### Strategy
1. Train initial MLIP on 140 structures
2. Run short MLIP-MD on diverse structures
3. Identify high-uncertainty configurations
4. Select 24 new structures for DFT
5. Retrain MLIP

### Uncertainty Quantification
- Ensemble variance from 5 models
- Force/energy prediction disagreement
- Diversity sampling with farthest-point

### Success Criteria
- Model improvement after AL iteration
- Final model meets accuracy targets

### Artifacts
- `mlip/results/al_iterations.json`
- `dft/results/al_selected_structures/`

---

## Phase 7: Validation & Analysis (Days 18-21)

### Tasks
- [ ] Compare MLIP vs DFT on test set
- [ ] Analyze structural parameters vs x
- [ ] Run MLIP-MD and validate stability
- [ ] Generate publication-ready plots

### Validation Metrics
- Energy RMSE
- Forces RMSE
- Lattice parameters comparison
- Fe-O bond lengths
- Octahedral tilts

### Success Criteria
- MLIP reproduces DFT trends
- MD simulations stable at 300K
- All trends documented

### Artifacts
- `analysis/results/validation_report.json`
- `analysis/results/figures/`
- `analysis/results/final_report.pdf`

---

## Phase 8: Documentation & Finalization (Days 21-22)

### Tasks
- [ ] Write final report
- [ ] Document all parameters
- [ ] Create reproducibility checklist
- [ ] Archive all data

### Artifacts
- Final report
- Complete parameter log
- Reproducibility guide

---

## Potential Bottlenecks and Mitigations

| Bottleneck | Impact | Mitigation |
|------------|--------|------------|
| GPU memory (16GB) | Limits batch size, model size | Use gradient checkpointing, smaller batches |
| WSL I/O | Slow file access | Keep data on Linux filesystem |
| GPAW GPU scaling | May not benefit small cells | Use CPU for Tier B, GPU for Tier A |
| DFT wall time | Long calculations | Use reasonable k-points, parallelize |
| MLIP overfitting | Poor generalization | Strong regularization, early stopping |

---

## File Inventory

```
├── README.md                    # Project overview
├── environment.yml              # Conda environment
├── project_plan.md              # This file
├── cifs/
│   ├── h-LuFeO3.cif            # Undoped reference
│   └── LSFO.cif                # x=0.5 reference
├── dft/
│   ├── config/
│   │   ├── gpaw_params.py      # DFT parameters
│   │   └── dft_budget.py       # Budget tracking
│   ├── scripts/
│   │   ├── structure_generation.py
│   │   ├── tier_a_relaxation.py
│   │   ├── tier_b_calculation.py
│   │   ├── run_dft_pipeline.py
│   │   └── extract_dft_data.py
│   ├── structures/
│   │   └── *.cif               # Generated structures
│   └── results/
│       ├── *.traj              # ASE trajectories
│       └── dft_dataset.json    # Consolidated data
├── mlip/
│   ├── config/
│   │   ├── model_config.py
│   │   └── training_config.py
│   ├── scripts/
│   │   ├── prepare_dataset.py
│   │   ├── train_mlip.py
│   │   ├── active_learning.py
│   │   └── validate_mlip.py
│   ├── models/
│   │   └── mace_model.pt
│   └── results/
│       └── *.json
├── analysis/
│   ├── scripts/
│   │   ├── analyze_structure.py
│   │   ├── compare_dft_mlip.py
│   │   └── plot_results.py
│   └── results/
│       └── figures/
└── docs/
    └── installation_log.txt
```

---

## Timeline Summary

| Phase | Duration | Days |
|-------|----------|------|
| Environment Setup | 2 days | 1-2 |
| Structure Generation | 1 day | 2-3 |
| DFT Calculations | 7 days | 3-10 |
| Dataset Preparation | 1 day | 10 |
| MLIP Training | 4 days | 10-14 |
| Active Learning | 4 days | 14-18 |
| Validation & Analysis | 3 days | 18-21 |
| Documentation | 1 day | 21-22 |
| **Total** | **22 days** | |

---

## References

1. Hexagonal phase stabilization and magnetic orders of multiferroic Lu₁₋ₓScₓFeO₃
2. Optimal Autonomous MLIP Dataset Building (Nature Computational Science, 2024)
3. GPAW Documentation: https://gpaw.readthedocs.io/
4. MACE: https://github.com/ACEsuit/mace