## 5. Archive (Pre-GaN / Oxide Workflow)

This project previously targeted Lu/Sc/Fe-oxide. Those commands are intentionally removed from the active plan to avoid confusion.

```bash
# A5) If you need to stop the pilot
# Press Ctrl+C in terminal running A1
```

### Stage B: Tier-B Mini Batch

Objective: Produce the smallest dataset that still runs extraction + MLIP training end-to-end.

Current execution checkpoint:
1. `Stage A`: done
2. `B1`: done
3. `B2`: in progress (run -> monitor -> plot)
4. Continue to `Stage D` only after `B2-plot` is completed
5. If completed Tier-B entries are still too few (<8), run `B2b` before `Stage D`

Success criteria:

1. 3 single-points complete (energies + forces).
2. Optional: 1-2 short-relax complete (adds off-equilibrium force data).
3. Outputs preserved and usable by `extract_data` + `train_mlip`.

Commands (each run must be followed by monitor + plot):

```bash
# B0) Activate environment
conda activate mlip_env
```

```bash
# B1) GPU mini-batch: 3 structures, single-point only (minimum-load)
python dft/scripts/tier_b_calculations.py --gpu --calc-type single_point --composition 0.5 --max-structures 3 --maxiter 25 --conv-energy 1e-3 --conv-density 1e-2 --conv-eigenstates 1e-4 --mag-config afm
```

```bash
# B1-monitor) Monitor latest SP/SR logs
LOG_SP="$(ls -t dft/results/logs/gpaw_tierb_sp__*.out 2>/dev/null | head -n 1)"; [ -z "$LOG_SP" ] && LOG_SP="dft/results/logs/gpaw_tierb_sp.out"; tail -f "$LOG_SP"
```

```bash
# B1-monitor2) Optional SR monitor
LOG_SR="$(ls -t dft/results/logs/gpaw_tierb_sr__*.out 2>/dev/null | head -n 1)"; [ -z "$LOG_SR" ] && LOG_SR="dft/results/logs/gpaw_tierb_sr.out"; tail -f "$LOG_SR"
```

```bash
# B1-plot) Plot latest SP and SR convergence after run
LOG_SP="$(ls -t dft/results/logs/gpaw_tierb_sp__*.out 2>/dev/null | head -n 1)"; [ -z "$LOG_SP" ] && LOG_SP="dft/results/logs/gpaw_tierb_sp.out"; python analysis/scripts/plot_scf_convergence.py --log "$LOG_SP" --out analysis/results/scf_convergence_B1_sp.png --csv analysis/results/scf_convergence_B1_sp.csv --title "B1 Tier-B SP Convergence"; LOG_SR="$(ls -t dft/results/logs/gpaw_tierb_sr__*.out 2>/dev/null | head -n 1)"; [ -z "$LOG_SR" ] && LOG_SR="dft/results/logs/gpaw_tierb_sr.out"; [ -f "$LOG_SR" ] && python analysis/scripts/plot_scf_convergence.py --log "$LOG_SR" --out analysis/results/scf_convergence_B1_sr.png --csv analysis/results/scf_convergence_B1_sr.csv --title "B1 Tier-B SR Convergence"
```

```bash
# B2) GPU short-relax on 1-2 structures (required before Stage D)
python dft/scripts/tier_b_calculations.py --gpu --calc-type short_relax --composition 0.5 --max-structures 2 --maxiter 25 --conv-energy 1e-3 --conv-density 1e-2 --conv-eigenstates 1e-4 --fmax 0.5 --relax-steps 8 --mag-config afm
```

```bash
# B2-monitor) Monitor B2 SR convergence log
LOG_SR="$(ls -t dft/results/logs/gpaw_tierb_sr__*.out 2>/dev/null | head -n 1)"; [ -z "$LOG_SR" ] && LOG_SR="dft/results/logs/gpaw_tierb_sr.out"; tail -f "$LOG_SR"
```

```bash
# B2-plot) Plot B2 SR convergence
LOG_SR="$(ls -t dft/results/logs/gpaw_tierb_sr__*.out 2>/dev/null | head -n 1)"; [ -z "$LOG_SR" ] && LOG_SR="dft/results/logs/gpaw_tierb_sr.out"; python analysis/scripts/plot_scf_convergence.py --log "$LOG_SR" --out analysis/results/scf_convergence_B2_sr.png --csv analysis/results/scf_convergence_B2_sr.csv --title "B2 Tier-B SR Convergence"
```

```bash
# B2-check) Check number of completed Tier-B entries
python -c "import json; d=json.load(open('dft/results/tier_b_results.json')); sp=len([x for x in d.get('single_point',[]) if x.get('status')=='completed']); sr=len([x for x in d.get('short_relax',[]) if x.get('status')=='completed']); print(f'completed_sp={sp} completed_sr={sr} total={sp+sr}')"
```

```bash
# B2b) Add more data if total<8 (single-point, x=0.2 and x=0.3, 2 each)
python dft/scripts/tier_b_calculations.py --gpu --calc-type single_point --composition 0.2 0.3 --max-structures 4 --maxiter 25 --conv-energy 1e-3 --conv-density 1e-2 --conv-eigenstates 1e-4 --mag-config afm
```

```bash
# B2b-monitor) Monitor B2b SP log
LOG_SP="$(ls -t dft/results/logs/gpaw_tierb_sp__*.out 2>/dev/null | head -n 1)"; [ -z "$LOG_SP" ] && LOG_SP="dft/results/logs/gpaw_tierb_sp.out"; tail -f "$LOG_SP"
```

```bash
# B2b-plot) Plot B2b SP convergence
LOG_SP="$(ls -t dft/results/logs/gpaw_tierb_sp__*.out 2>/dev/null | head -n 1)"; [ -z "$LOG_SP" ] && LOG_SP="dft/results/logs/gpaw_tierb_sp.out"; python analysis/scripts/plot_scf_convergence.py --log "$LOG_SP" --out analysis/results/scf_convergence_B2b_sp.png --csv analysis/results/scf_convergence_B2b_sp.csv --title "B2b Tier-B SP Convergence"
```

If GPU is unavailable and you must use CPU, do it explicitly with MPI (8 or 12 ranks). Example (CPU MPI-12, single-point only, minimal):

```bash
MLIP_MPI_PROCS=12 python dft/scripts/tier_b_calculations.py --calc-type single_point --composition 0.5 --max-structures 3 --maxiter 25 --conv-energy 1e-3 --conv-density 1e-2 --conv-eigenstates 1e-4 --mag-config afm
```

```bash
# B3) Monitor CPU utilization
top
```

```bash
# B4) Monitor progress/results file
tail -f dft/results/tier_b_results.json
```

### Stage C: Full Tier-B Run (Optional)

Objective: Only if you need a larger dataset for a higher-quality model.

For the pipeline demo, Stage C is not required.

### Section Transition (Your Current State)

If `Stage B` is finished and you intentionally skip `Stage C`, the next section is **Stage D**.

Next experiments (in order):

```bash
# T1) Extract data from completed Tier-B
python run_pipeline.py --stages extract_data
```

```bash
# T2) Train MLIP (with profiler pre-check)
python mlip/scripts/train_mlip.py --profile-before-train --profile-steps 20
```

```bash
# T3) Run analysis and validation
python run_pipeline.py --stages analysis validation
```

### Stage D: Extract + Train + Analyze + Validate

Objective: Run minimum MLIP experiments after DFT data collection.

Success criteria:
1. `extract_data` creates non-trivial `train.xyz` and `all_data.xyz`.
2. training starts on CUDA and writes run logs/artifacts.
3. analysis/validation stage finishes.

#### D0) Extract dataset
Why: convert Tier-B outputs to MLIP-ready extxyz files.

```bash
# D0-run
python run_pipeline.py --stages extract_data
```

```bash
# D0-monitor
watch -n 2 "ls -lh mlip/data/*.xyz 2>/dev/null; echo '---'; cat mlip/data/dataset_stats.json 2>/dev/null"
```

```bash
# D0-plot/check (must pass before D1/D2)
python -c "from pathlib import Path; from mlip.scripts.train_mlip import count_structures; t=count_structures(Path('mlip/data/train.xyz')) if Path('mlip/data/train.xyz').exists() else 0; v=count_structures(Path('mlip/data/val.xyz')) if Path('mlip/data/val.xyz').exists() else 0; a=count_structures(Path('mlip/data/all_data.xyz')) if Path('mlip/data/all_data.xyz').exists() else 0; print(f'train={t} val={v} all={a}'); assert a>=8, 'Need more Tier-B data: return to Stage B2b'"
```

#### D1) Profiler-only probe
Why: validate data-loader/GPU pipeline before long training.

```bash
# D1-run
python mlip/scripts/train_mlip.py --profile-only --profile-steps 20
```

```bash
# D1-monitor
watch -n 1 nvidia-smi
```

```bash
# D1-plot/check
RUN="$(ls -td mlip/results/mace_run_* 2>/dev/null | head -n 1)"; echo "$RUN"; ls -lh "$RUN"/profiler/torch_profile_trace.json "$RUN"/profiler/torch_profile_summary.txt
```

#### D2) Train MLIP
Why: produce a first GPU-trained MACE model from current dataset.

```bash
# D2-run
python mlip/scripts/train_mlip.py --profile-before-train --profile-steps 20
```

```bash
# D2-monitor
watch -n 1 nvidia-smi
```

```bash
# D2-monitor2
RUN="$(ls -td mlip/results/mace_run_* 2>/dev/null | head -n 1)"; tail -f "$RUN"/mace_stdout.log
```

```bash
# D2-plot/check
RUN="$(ls -td mlip/results/mace_run_* 2>/dev/null | head -n 1)"; python -c "import json, pathlib; p=pathlib.Path('$RUN')/'epoch_device_log.jsonl'; rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []; print('epochs_logged=',len(rows)); print('last_epoch=',rows[-1]['epoch'] if rows else None); print('device=',rows[-1]['device'] if rows else None)"
```

Troubleshooting for D2:
1. If you see `None of 'REF_energy', 'REF_forces'...`, rerun with latest `train_mlip.py` (keys are forced to `energy/forces/stress`).
2. If you see `assert size > 1`, dataset is too small; run `B2b`, then rerun `D0`.

#### D3) Analysis and validation
Why: complete pipeline and produce final quality checks.

```bash
# D3-run
python run_pipeline.py --stages analysis validation
```

```bash
# D3-monitor
tail -f analysis/results/*.log 2>/dev/null
```

```bash
# D3-plot/check
ls -lt analysis/results mlip/results
```

## 4. Command Logging Rules

1. Log every user-run command before/after execution.
2. Record result status: `pending`, `running`, `success`, `failed`, `stopped`.
3. Add short notes (runtime, errors, convergence behavior).
4. For every command, include continuity note: `input_from=<previous step/artifact>`.
5. If `--reset-state` is used, add `reset_reason=<why mandatory>` and treat it as workflow break.

## 5. Command Log

| # | Timestamp (local) | Command | Stage | Status | Notes |
|---|---|---|---|---|---|
| 1 | 2026-02-28 22:xx | `conda activate mlip_env` | A | success | Environment prepared. |
| 2 | 2026-02-28 22:xx | `python dft/scripts/tier_b_calculations.py --gpu --calc-type single_point --composition 0.5 --max-structures 1 --maxiter 30 --conv-energy 1e-4 --conv-density 1e-3 --conv-forces 0.1 --conv-eigenstates 1e-5 --mag-config afm` | A | pending | Mandatory full-args pilot for x=0.5. |
| 3 | 2026-02-28 22:xx | `watch -n 1 nvidia-smi` | A | success | Confirmed GPU utilization at 100%. |
| 4 | 2026-02-28 22:xx | `tail -f dft/results/logs/gpaw_tierb_sp.out` | A | success | Observed SCF progress. |
| 5 | pending | `MLIP_MPI_PROCS=8 python run_pipeline.py --no-gpu --stages dft_tier_b` | B | pending | CPU mini-batch throughput test (MPI-8). |
| 6 | pending | `MLIP_MPI_PROCS=12 python run_pipeline.py --no-gpu --stages dft_tier_b` | B/C | pending | Full Tier-B via pipeline (CPU MPI). |
| 7 | pending | `python run_pipeline.py --stages dft_tier_b` | B/C | pending | GPU Tier-B execution (no single-CPU mode). |
| 8 | pending | `python run_pipeline.py --stages extract_data train_mlip analysis validation` | D | pending | Execute after enough completed DFT structures exist. |
| 9 | 2026-03-01 08:xx | `python dft/scripts/tier_b_calculations.py --reset-state --dry-run --calc-type single_point --composition 0.5 --max-structures 1 --maxiter 30 --conv-energy 1e-4 --conv-density 1e-3 --conv-eigenstates 1e-5 --mag-config afm` | A | success | Tier-B reset completed; removed prior Tier-B outputs/bookkeeping and set counts back to zero. |
| 10 | 2026-03-01 08:xx | `python -c "import os; from ase.io import read; from dft.config.gpaw_params import get_kpts_for_supercell; a=read('dft/structures/Lu0.5Sc0.5FeO3_config00.cif'); c=a.get_cell(); b=5.96; s=(max(1,round(c[0,0]/b)),max(1,round(c[1,1]/b)),max(1,round(c[2,2]/(2*b)))); print(f'supercell={s} kpts={get_kpts_for_supercell(s,(4,4,2))} ecut_eV={os.environ.get(\"MLIP_GPAW_GPU_ECUT\",\"350.0\")}')"` | A | success | Captured metadata: `supercell=(2, 2, 1)`, `kpts=(2, 2, 2)`, `ecut_eV=350.0`. |
| 11 | 2026-03-01 09:xx | `python -c "import json; from pathlib import Path; p=Path('dft/results/dft_budget.json'); s=json.loads(p.read_text()); calc=s.get('calculations',[]); new=[]; seen=set(); [new.append(c) if not ((c.get('type'),c.get('structure'),c.get('status')) in seen and c.get('status')=='completed') else None or seen.add((c.get('type'),c.get('structure'),c.get('status'))) for c in calc]; s['calculations']=new; s['tier_a_used']=sum(1 for c in new if c.get('type')=='tier_a' and c.get('status')=='completed'); s['tier_b_sp_used']=sum(1 for c in new if c.get('type')=='tier_b_sp' and c.get('status')=='completed'); s['tier_b_relax_used']=sum(1 for c in new if c.get('type')=='tier_b_relax' and c.get('status')=='completed'); s['active_learning_used']=sum(1 for c in new if c.get('type')=='active_learning' and c.get('status')=='completed'); s['total_used']=s['tier_a_used']+s['tier_b_sp_used']+s['tier_b_relax_used']+s['active_learning_used']; p.write_text(json.dumps(s, indent=2)); print('budget_dedup_done')"` | A | success | Removed duplicate Tier-B count for repeated continuation of same structure; budget returned to `total=1`, `SP=1`. |
| 12 | 2026-03-01 10:xx | `python dft/scripts/tier_b_calculations.py --gpu --use-restart --calc-type short_relax --composition 0.5 --max-structures 1 --maxiter 30 --conv-energy 1e-4 --conv-density 1e-3 --conv-eigenstates 1e-5 --conv-forces 0.2 --mag-config afm` | A1b | success | Force-minimization result: `initial_energy=-676.368681 eV`, `final_energy=-683.567340 eV`, `deltaE=-7.198659 eV`, `final_max_force=0.182812 eV/Ang` (met `fmax=0.2`), `elapsed=3034.76 s (~50.6 min)`, `GPU=True`. Output traj: `dft/results/trajectories/tierb_short_relax_Lu0.5Sc0.5FeO3_config00.traj`. |
| 13 | 2026-03-02 10:xx | `python dft/scripts/tier_b_calculations.py --reset-state --dry-run --calc-type single_point --composition 0.5 --max-structures 1 --mag-config afm` | A | success | Emergency reset (workflow break). `reset_reason=stopped run cleanup`; removed Tier-B residue files and reset budget back to zero. |
