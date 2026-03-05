"""
Tier A DFT Calculations (GaN): Minimal Full Relaxations
=======================================================

Tier A is optional in the GaN minimal-load workflow. If you run it, keep it
small: 1 primitive bulk and (optionally) 1 small-defect cell.

By default, this script:
- selects a small number of structures from dft/structures/structure_info.json
- performs ionic-only relaxation (fixed cell) with loose thresholds

Rationale:
- For an MLIP pipeline demo, Tier-B (single-points + short relax) is usually enough.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS

from ase.parallel import parprint

try:
    from gpaw import GPAW, PW
    from gpaw.mixer import Mixer
    from gpaw.occupations import FermiDirac
    GPAW_AVAILABLE = True
except Exception:
    GPAW_AVAILABLE = False

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.gpaw_params import (
    GPAW_PARAMS, setup_gpu_environment, get_kpts_for_supercell, LATTICE_PARAMS
)
from config.dft_budget import DFTBudgetTracker


PROJECT_ROOT = Path(__file__).parent.parent.parent
STRUCTURES_DIR = PROJECT_ROOT / "dft" / "structures"
RESULTS_DIR = PROJECT_ROOT / "dft" / "results"
LOGS_DIR = RESULTS_DIR / "logs"
TRAJECTORY_DIR = RESULTS_DIR / "trajectories"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


def _infer_supercell_size(atoms):
    base = LATTICE_PARAMS.get("GaN", {})
    a0 = float(base.get("a", 3.2))
    c0 = float(base.get("c", 5.2))
    lengths = atoms.get_cell().lengths()
    nx = max(1, int(round(float(lengths[0]) / a0)))
    ny = max(1, int(round(float(lengths[1]) / a0)))
    nz = max(1, int(round(float(lengths[2]) / c0)))
    return (nx, ny, nz)


def create_gpaw_calculator(supercell_size, gpu=True, txt_path=None, ecut_eV=350.0):
    if not GPAW_AVAILABLE:
        return None

    params = GPAW_PARAMS.copy()
    params.pop("basis", None)

    # occupations
    sigma = params.pop("sigma", 0.05)
    occ = params.get("occupations", None)
    if isinstance(occ, str) and occ.lower() == "smearing":
        params["occupations"] = FermiDirac(width=float(sigma))

    # mixer
    mixer_cfg = params.get("mixer")
    if isinstance(mixer_cfg, dict):
        params["mixer"] = Mixer(beta=mixer_cfg.get("beta", 0.1), nmaxold=mixer_cfg.get("nmaxold", 5))

    params["kpts"] = get_kpts_for_supercell(supercell_size, base_kpts=tuple(params.get("kpts", (6, 6, 4))))
    params["spinpol"] = False
    params["txt"] = str(txt_path) if txt_path else str(LOGS_DIR / "gpaw_tier_a.out")

    if gpu:
        setup_gpu_environment()
        params["mode"] = PW(float(ecut_eV))
        params.pop("h", None)
        params["parallel"] = {"gpu": True}
    else:
        params["parallel"] = {"gpu": False}

    return GPAW(**params)


def load_structure_list(tags=None, max_structures=1):
    info_file = STRUCTURES_DIR / "structure_info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"Missing {info_file}; run dft/scripts/structure_generation.py first.")
    info = json.load(open(info_file, "r", encoding="utf-8"))
    structs = info.get("structures", [])
    wanted = set(tags or [])
    out = []
    for s in structs:
        stags = set(s.get("tags") or [])
        if wanted and stags.isdisjoint(wanted):
            continue
        out.append(s)
        if len(out) >= int(max_structures):
            break
    return out


def run_relax(struct_meta, gpu=True, fmax=0.5, steps=40):
    struct_file = struct_meta["filepath"]
    sid = struct_meta.get("id") or Path(struct_file).stem
    atoms = read(struct_file)
    sc = struct_meta.get("supercell_size") or _infer_supercell_size(atoms)

    name = f"tiera_relax_{sid}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    txt_path = LOGS_DIR / f"gpaw_tiera__{datetime.now().strftime('%Y%m%d_%H%M%S')}.out"
    calc = create_gpaw_calculator(sc, gpu=gpu, txt_path=txt_path)
    if calc is None:
        return {"name": name, "status": "skipped", "reason": "GPAW not available", "structure_id": sid}
    atoms.calc = calc

    t0 = time.time()
    e0 = atoms.get_potential_energy()
    f0 = np.abs(atoms.get_forces()).max()

    opt_log = LOGS_DIR / f"{name}_opt.log"
    opt = LBFGS(atoms, logfile=str(opt_log))
    opt.run(fmax=float(fmax), steps=int(steps))

    e1 = atoms.get_potential_energy()
    f1 = np.abs(atoms.get_forces()).max()
    elapsed = time.time() - t0

    traj_path = TRAJECTORY_DIR / f"{name}.traj"
    atoms_to_save = atoms.copy()
    atoms_to_save.calc = None
    write(traj_path, atoms_to_save)

    # checkpoint
    gpw_path = CHECKPOINT_DIR / f"{name}.gpw"
    try:
        atoms.calc.write(str(gpw_path), mode="all")
    except TypeError:
        atoms.calc.write(str(gpw_path))

    return {
        "name": name,
        "status": "completed",
        "structure_id": sid,
        "tags": struct_meta.get("tags"),
        "n_atoms": int(len(atoms)),
        "initial_energy": float(e0),
        "final_energy": float(e1),
        "initial_max_force": float(f0),
        "final_max_force": float(f1),
        "elapsed_time": float(elapsed),
        "trajectory_file": str(traj_path),
        "checkpoint_file": str(gpw_path),
        "used_gpu": bool(gpu),
        "txt": str(txt_path),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Tier-A minimal relaxations for GaN")
    p.add_argument("--gpu", action="store_true", help="Use GPU (requires real CuPy backend).")
    p.add_argument("--tags", nargs="+", default=["bulk"], help="Select structures by tags.")
    p.add_argument("--max-structures", type=int, default=1, help="Max structures to relax.")
    p.add_argument("--fmax", type=float, default=0.5, help="Ionic relaxation threshold (eV/Ang).")
    p.add_argument("--steps", type=int, default=40, help="Max optimizer steps.")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = DFTBudgetTracker(RESULTS_DIR / "dft_budget.json")
    tracker.print_status()

    selected = load_structure_list(tags=args.tags, max_structures=args.max_structures)
    print(f"Selected {len(selected)} structures for Tier-A relaxation.")
    results = {"tier": "A", "system": "GaN", "started": datetime.now().isoformat(), "calculations": []}

    for s in selected:
        if not tracker.can_run("tier_a"):
            parprint("Tier-A budget exhausted; stopping.")
            break
        r = run_relax(s, gpu=bool(args.gpu), fmax=args.fmax, steps=args.steps)
        results["calculations"].append(r)
        if r.get("status") == "completed":
            tracker.record_calculation("tier_a", r["name"], composition=None, energy=r.get("final_energy"), forces_max=r.get("final_max_force"))

        out_file = RESULTS_DIR / "tier_a_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    results["completed"] = datetime.now().isoformat()
    out_file = RESULTS_DIR / "tier_a_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()

