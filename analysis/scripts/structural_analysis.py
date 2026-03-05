"""
Structural analysis for workflow outputs (GaN pivot).

This script summarizes the generated structures and DFT completion status.
It supports both:
- legacy oxide schema (structure_info.json has "compositions")
- GaN schema (structure_info.json has "structures" with "tags")
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import read


PROJECT_ROOT = Path(__file__).parent.parent.parent
STRUCTURES_DIR = PROJECT_ROOT / "dft" / "structures"
RESULTS_DIR = PROJECT_ROOT / "dft" / "results"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "results"


def _safe_read(path: Path):
    try:
        atoms = read(path)
        if isinstance(atoms, list):
            atoms = atoms[-1]
        return atoms
    except Exception:
        return None


def analyze_structures():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    info_file = STRUCTURES_DIR / "structure_info.json"
    if not info_file.exists():
        print(f"structure_info.json not found at {info_file}")
        return {}

    with open(info_file, "r") as f:
        info = json.load(f)

    per_comp = {}
    by_tag = {}
    by_supercell = {}

    # Legacy oxide schema
    for x, comp in info.get("compositions", {}).items():
        files = [Path(s["filepath"]) for s in comp.get("structures", [])]
        n_atoms = []
        volumes = []
        for fpath in files:
            atoms = _safe_read(fpath)
            if atoms is None:
                continue
            n_atoms.append(len(atoms))
            volumes.append(float(atoms.get_volume()))
        per_comp[x] = {
            "n_structures": int(len(n_atoms)),
            "mean_atoms": float(np.mean(n_atoms)) if n_atoms else 0.0,
            "mean_volume": float(np.mean(volumes)) if volumes else 0.0,
            "std_volume": float(np.std(volumes)) if volumes else 0.0,
        }

    # GaN schema (tag-based)
    for s in info.get("structures", []):
        fpath = Path(s["filepath"])
        atoms = _safe_read(fpath)
        if atoms is None:
            continue
        n = int(len(atoms))
        v = float(atoms.get_volume())
        sc = tuple(s.get("supercell_size") or ())
        tags = s.get("tags") or []

        if sc:
            rec = by_supercell.setdefault(str(sc), {"n_structures": 0, "mean_atoms": 0.0, "mean_volume": 0.0})
            rec["n_structures"] += 1
            rec["mean_atoms"] += n
            rec["mean_volume"] += v

        for t in tags:
            rec = by_tag.setdefault(str(t), {"n_structures": 0, "mean_atoms": 0.0, "mean_volume": 0.0})
            rec["n_structures"] += 1
            rec["mean_atoms"] += n
            rec["mean_volume"] += v

    for rec in by_tag.values():
        if rec["n_structures"]:
            rec["mean_atoms"] = float(rec["mean_atoms"] / rec["n_structures"])
            rec["mean_volume"] = float(rec["mean_volume"] / rec["n_structures"])
    for rec in by_supercell.values():
        if rec["n_structures"]:
            rec["mean_atoms"] = float(rec["mean_atoms"] / rec["n_structures"])
            rec["mean_volume"] = float(rec["mean_volume"] / rec["n_structures"])

    tier_a = {}
    tier_a_file = RESULTS_DIR / "tier_a_results.json"
    if tier_a_file.exists():
        with open(tier_a_file, "r") as f:
            data = json.load(f)
        completed = [c for c in data.get("calculations", []) if c.get("status") == "completed"]
        tier_a = {
            "n_completed": len(completed),
            "energies": [c.get("final_energy") for c in completed if c.get("final_energy") is not None],
        }

    tier_b = {}
    tier_b_file = RESULTS_DIR / "tier_b_results.json"
    if tier_b_file.exists():
        with open(tier_b_file, "r") as f:
            data = json.load(f)
        sp_ok = [c for c in data.get("single_point", []) if c.get("status") == "completed"]
        sr_ok = [c for c in data.get("short_relax", []) if c.get("status") == "completed"]
        tier_b = {
            "single_point_completed": len(sp_ok),
            "short_relax_completed": len(sr_ok),
        }

    summary = {
        "created": datetime.now().isoformat(),
        "system": info.get("system"),
        "base_cif": info.get("base_cif"),
        "per_composition": per_comp,
        "by_tag": by_tag,
        "by_supercell_size": by_supercell,
        "tier_a": tier_a,
        "tier_b": tier_b,
    }

    out = ANALYSIS_DIR / "structural_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("Structural analysis complete")
    print(f"Saved: {out}")
    return summary


if __name__ == "__main__":
    analyze_structures()
