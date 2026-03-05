"""
Extract DFT Data for MLIP Training
===================================
Convert Tier A / Tier B outputs into extxyz datasets for MLIP training.

This script is intentionally conservative:
- By default it extracts *all* completed Tier-A/Tier-B results.
- Optionally, you can filter by maximum atom count (e.g. to keep training light and
  use large-supercell DFT only as "spot-check" gates).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


PROJECT_ROOT = Path(__file__).parent.parent.parent
DFT_RESULTS_DIR = PROJECT_ROOT / "dft" / "results"
MLIP_DATA_DIR = PROJECT_ROOT / "mlip" / "data"
MLIP_DATASETS_DIR = MLIP_DATA_DIR / "datasets"


def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: Failed to read {path}: {e!r}")
        return {}


def _read_atoms_from_result(result: dict):
    """Try trajectory file first, then original structure file."""
    candidate_paths = [result.get("trajectory_file"), result.get("structure_file")]
    for p in candidate_paths:
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            continue
        try:
            atoms = read(path)
            if isinstance(atoms, list):
                atoms = atoms[-1]
            return atoms
        except Exception:
            continue
    return None


def _normalize_result_entry(result: dict, source: str):
    """Build a unified entry from one DFT result block."""
    if result.get("status") != "completed":
        return None

    energy = result.get("final_energy")
    if energy is None:
        energy = result.get("energy")
    if energy is None:
        return None

    atoms = _read_atoms_from_result(result)
    if atoms is None:
        return None

    return {
        "name": result.get("name", "unknown"),
        "source": source,
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.get_cell().tolist(),
        "pbc": atoms.get_pbc().tolist(),
        "atomic_numbers": atoms.get_atomic_numbers().tolist(),
        "energy": float(energy),
        "forces": result.get("forces"),
        "stress": result.get("stress"),
    }


def _to_atoms(entry: dict):
    atoms = Atoms(
        numbers=entry["atomic_numbers"],
        positions=entry["positions"],
        cell=entry["cell"],
        pbc=entry.get("pbc", [True, True, True]),
    )

    calc_kwargs = {"energy": float(entry["energy"])}
    if entry.get("forces") is not None:
        calc_kwargs["forces"] = np.array(entry["forces"])
    if entry.get("stress") is not None:
        calc_kwargs["stress"] = np.array(entry["stress"])

    atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    return atoms


def _save_xyz(atoms_list, output_path: Path):
    if not atoms_list:
        return
    write(output_path, atoms_list, format="extxyz")
    print(f"  Saved {len(atoms_list)} -> {output_path}")


def extract_all_dft_data(max_atoms: int | None = None):
    print("\n" + "=" * 60)
    print("EXTRACTING DFT DATA FOR MLIP TRAINING")
    print("=" * 60)

    MLIP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MLIP_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = MLIP_DATASETS_DIR / f"dataset_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tier_a = _load_json(DFT_RESULTS_DIR / "tier_a_results.json")
    tier_b = _load_json(DFT_RESULTS_DIR / "tier_b_results.json")

    entries = []

    for result in tier_a.get("calculations", []):
        entry = _normalize_result_entry(result, source="tier_a")
        if entry is not None:
            entries.append(entry)

    for result in tier_b.get("single_point", []):
        entry = _normalize_result_entry(result, source="tier_b_sp")
        if entry is not None:
            entries.append(entry)

    for result in tier_b.get("short_relax", []):
        entry = _normalize_result_entry(result, source="tier_b_sr")
        if entry is not None:
            entries.append(entry)

    # Deduplicate by (source, name)
    dedup = {}
    for e in entries:
        dedup[(e["source"], e["name"])] = e
    entries = list(dedup.values())

    print(f"\nCollected completed DFT entries: {len(entries)}")

    dataset_json = out_dir / "dft_dataset.json"
    stats_json = out_dir / "dataset_stats.json"
    latest_ptr = MLIP_DATA_DIR / "LATEST_DATASET.txt"

    if not entries:
        with open(dataset_json, "w") as f:
            json.dump([], f, indent=2)

        stats = {
            "total_structures": 0,
            "train": 0,
            "val": 0,
            "test": 0,
            "sources": {"tier_a": 0, "tier_b_sp": 0, "tier_b_sr": 0},
            "created": datetime.now().isoformat(),
        }
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2)

        print("No completed DFT data yet. Wrote empty metadata files.")
        print(f"  {dataset_json}")
        print(f"  {stats_json}")
        return []

    atoms_list_full = [_to_atoms(e) for e in entries]

    # Optional filter: keep training datasets small (fast MLIP training).
    if max_atoms is not None:
        max_atoms = int(max_atoms)
        atoms_list = [a for a in atoms_list_full if len(a) <= max_atoms]
    else:
        atoms_list = atoms_list_full

    n_total = len(atoms_list)
    n_total_full = len(atoms_list_full)

    # Small-dataset policy (pipeline demo):
    # - Ensure val set exists when n_total >= 3 so MACE can train without internal split assertions.
    # - Prefer: train=n_total-1, val=1, test=0 for n_total <= 4.
    if n_total <= 2:
        n_val = 0
        n_train = n_total
    elif n_total <= 4:
        n_val = 1
        n_train = n_total - n_val
    else:
        n_train = int(0.8 * n_total)
        n_val = max(1, int(0.1 * n_total))
        if n_train + n_val > n_total:
            n_val = max(1, n_total - n_train)

    np.random.seed(42)
    idx = np.random.permutation(n_total)

    train_atoms = [atoms_list[i] for i in idx[:n_train]]
    val_atoms = [atoms_list[i] for i in idx[n_train:n_train + n_val]]
    test_atoms = [atoms_list[i] for i in idx[n_train + n_val:]]

    _save_xyz(train_atoms, out_dir / "train.xyz")
    _save_xyz(val_atoms, out_dir / "val.xyz")
    _save_xyz(test_atoms, out_dir / "test.xyz")
    _save_xyz(atoms_list, out_dir / "all_data.xyz")
    # Always preserve the full, unfiltered dataset for bookkeeping/debugging.
    _save_xyz(atoms_list_full, out_dir / "all_data_full.xyz")

    # Persist *filtered* entries for training, plus a full copy for traceability.
    if max_atoms is not None:
        # Rebuild filtered entries by natoms to match atoms_list.
        # This keeps metadata consistent with train/val/test/all_data.
        filtered_entries = [e for e in entries if len(e.get("atomic_numbers", [])) <= max_atoms]
    else:
        filtered_entries = entries

    with open(dataset_json, "w") as f:
        json.dump(filtered_entries, f, indent=2)

    with open(out_dir / "dft_dataset_full.json", "w") as f:
        json.dump(entries, f, indent=2)

    stats = {
        "total_structures": n_total,
        "total_structures_full": n_total_full,
        "max_atoms_filter": int(max_atoms) if max_atoms is not None else None,
        "train": len(train_atoms),
        "val": len(val_atoms),
        "test": len(test_atoms),
        "sources": {
            "tier_a": len([e for e in filtered_entries if e["source"] == "tier_a"]),
            "tier_b_sp": len([e for e in filtered_entries if e["source"] == "tier_b_sp"]),
            "tier_b_sr": len([e for e in filtered_entries if e["source"] == "tier_b_sr"]),
        },
        "created": datetime.now().isoformat(),
    }

    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2)

    # Update "latest" pointer for downstream scripts (training, analysis).
    with open(latest_ptr, "w", encoding="utf-8") as f:
        f.write(str(out_dir) + "\n")

    print("\n" + "=" * 60)
    print("DATA EXTRACTION COMPLETE")
    print("=" * 60)
    if max_atoms is None:
        print(f"Total structures: {n_total}")
    else:
        print(f"Total structures (filtered): {n_total} (max_atoms={max_atoms})")
        print(f"Total structures (full): {n_total_full}")
    print(f"Output directory: {out_dir}")
    print(f"Latest pointer: {latest_ptr}")

    return entries


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract Tier-A/Tier-B DFT results into extxyz datasets for MACE training")
    ap.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="If set, exclude configs with natoms > max_atoms from train/val/test/all_data (still write all_data_full.xyz).",
    )
    args = ap.parse_args()
    extract_all_dft_data(max_atoms=args.max_atoms)
