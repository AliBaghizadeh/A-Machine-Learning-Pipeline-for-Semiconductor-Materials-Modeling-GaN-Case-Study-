#!/usr/bin/env python3
"""
Force Gate: MLIP vs DFT (Local Defect Region)
=============================================

This script compares MLIP (MACE) forces to stored DFT forces from Tier-B results
for a specific structure_id, and evaluates a simple pass/fail criterion.

Why:
- Energy-only gates can hide local force failures (especially near defects).
- A localized force check is cheap (seconds) and improves reliability for PoC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_latest_model() -> Path:
    runs = sorted((PROJECT_ROOT / "mlip" / "results").glob("mace_run_*"))
    if not runs:
        raise FileNotFoundError("No mlip/results/mace_run_* directories found.")
    run = runs[-1]
    models = sorted((run / "checkpoints").glob("*.model"))
    if not models:
        raise FileNotFoundError(f"No *.model found in {run}/checkpoints")
    return models[-1]

def _read_atoms_from_result(result: dict):
    """
    Read the geometry corresponding to a Tier-B result entry.
    Prefer trajectory_file (final structure), fall back to structure_file.
    """
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
            return atoms, str(path)
        except Exception:
            continue
    return None, None


def _select_dft_row(dft_json: Path, structure_id: str, source: str):
    d = json.load(open(dft_json, "r"))
    sp = [x for x in d.get("single_point", []) if x.get("status") == "completed" and x.get("structure_id") == structure_id]
    sr = [x for x in d.get("short_relax", []) if x.get("status") == "completed" and x.get("structure_id") == structure_id]

    source = (source or "auto").lower()
    if source == "sp":
        rows = sp
        label = "tier_b_sp"
    elif source == "sr":
        rows = sr
        label = "tier_b_sr"
    else:
        # For force gating, a relaxed/end-of-run geometry is usually more meaningful.
        rows = sr if sr else sp
        label = "tier_b_sr" if sr else "tier_b_sp"

    if not rows:
        raise KeyError(f"No completed DFT entry found for structure_id={structure_id!r} (source={source!r}) in {dft_json}")
    row = rows[-1]
    forces = row.get("forces")
    if forces is None:
        raise KeyError(f"DFT entry for structure_id={structure_id!r} has no forces stored.")
    return np.asarray(forces, dtype=float), row, label


def _coordination_mask(atoms, rcut: float, max_coord: int) -> np.ndarray:
    """
    Identify likely defect-region atoms by low coordination to opposite species.
    For wurtzite GaN, ideal coordination is ~4. Atoms with <= max_coord neighbors
    within rcut are treated as "defect-like".
    """
    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    # Build neighbor distances with minimal-image convention.
    inv_cell = np.linalg.inv(cell.T)
    frac = (pos @ inv_cell)
    frac -= np.floor(frac)

    def mic(dr_frac):
        if pbc is None:
            return dr_frac
        out = dr_frac.copy()
        for a in range(3):
            if bool(pbc[a]):
                out[:, a] -= np.round(out[:, a])
        return out

    n = len(atoms)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        zi = nums[i]
        # Opposite species in GaN (Ga=31, N=7).
        target = 7 if zi == 31 else 31
        dr = frac - frac[i]
        dr = mic(dr)
        # Cartesian displacements
        dr_cart = dr @ cell
        dist = np.linalg.norm(dr_cart, axis=1)
        # exclude self and wrong species
        neigh = (dist > 1e-8) & (dist <= float(rcut)) & (nums == target)
        coord = int(np.count_nonzero(neigh))
        if coord <= int(max_coord):
            mask[i] = True

    return mask


def _force_stats(df: np.ndarray) -> dict:
    per_atom = np.linalg.norm(df, axis=1)
    return {
        "mae": float(np.mean(per_atom)),
        "rmse": float(np.sqrt(np.mean(per_atom**2))),
        "max": float(np.max(per_atom) if len(per_atom) else float("nan")),
        "n_atoms": int(len(per_atom)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="MLIP-vs-DFT force gate (optionally localized to defect region)")
    p.add_argument("--structure-id", required=True, help="structure_id in dft/results/tier_b_results.json")
    p.add_argument("--cif", default=None, help="CIF file path (default: dft/structures/<structure_id>.cif)")
    p.add_argument("--dft-json", default=str(PROJECT_ROOT / "dft" / "results" / "tier_b_results.json"))
    p.add_argument("--model", default=None, help="MACE *.model path (default: latest mlip/results/mace_run_*/checkpoints/*.model)")
    p.add_argument("--device", default="cuda", help="MACE device (default: cuda; use cpu for offline demo)")
    p.add_argument(
        "--source",
        choices=["auto", "sp", "sr"],
        default="auto",
        help="Which DFT forces to use from tier_b_results.json (default: auto; prefers short_relax if present).",
    )
    p.add_argument(
        "--use-dft-geometry",
        action="store_true",
        help="Use the geometry from the selected DFT result entry (trajectory_file/structure_file). Recommended for SR. (default: False)",
    )

    p.add_argument("--select", choices=["all", "coord"], default="coord", help="Atom selection for gate (default: coord)")
    p.add_argument("--coord-rcut", type=float, default=2.4, help="Coordination cutoff in Angstrom (default: 2.4)")
    p.add_argument("--coord-max", type=int, default=3, help="Max coordination to consider 'defect-like' (default: 3)")

    p.add_argument("--mae-thresh", type=float, default=0.25, help="Selected-region force MAE threshold (eV/A)")
    p.add_argument("--max-thresh", type=float, default=1.0, help="Selected-region force MAX threshold (eV/A)")
    args = p.parse_args()

    structure_id = args.structure_id
    cif = Path(args.cif) if args.cif else (PROJECT_ROOT / "dft" / "structures" / f"{structure_id}.cif")
    dft_json = Path(args.dft_json)
    model = Path(args.model) if args.model else _load_latest_model()

    f_dft, row, label = _select_dft_row(dft_json, structure_id, source=args.source)

    if args.use_dft_geometry:
        atoms, atoms_path = _read_atoms_from_result(row)
        if atoms is None:
            atoms = read(cif)
            atoms_path = str(cif)
    else:
        atoms = read(cif)
        atoms_path = str(cif)

    if len(f_dft) != len(atoms):
        raise ValueError(f"Force length mismatch: DFT forces={len(f_dft)} atoms={len(atoms)} for {structure_id}")

    atoms.calc = MACECalculator(model_paths=str(model), device=str(args.device))
    f_mlip = np.asarray(atoms.get_forces(), dtype=float)
    df = f_mlip - f_dft

    overall = _force_stats(df)

    if args.select == "all":
        mask = np.ones(len(atoms), dtype=bool)
    else:
        mask = _coordination_mask(atoms, rcut=float(args.coord_rcut), max_coord=int(args.coord_max))
        # Fallback if coordination heuristic finds nothing.
        if not np.any(mask):
            per_atom = np.linalg.norm(f_dft, axis=1)
            k = min(12, len(per_atom))
            idx = np.argsort(-per_atom)[:k]
            mask = np.zeros(len(atoms), dtype=bool)
            mask[idx] = True

    selected = _force_stats(df[mask])
    ok = (selected["mae"] <= float(args.mae_thresh)) and (selected["max"] <= float(args.max_thresh))

    print(f"structure_id={structure_id}")
    print(f"dft_source={label} (requested={args.source})")
    print(f"geometry={atoms_path}")
    print(f"cif_arg={cif}")
    print(f"model={model}")
    print(f"device={args.device}")
    print(f"selection={args.select} selected_atoms={selected['n_atoms']} total_atoms={overall['n_atoms']}")
    print(f"overall:  mae={overall['mae']:.6f} rmse={overall['rmse']:.6f} max={overall['max']:.6f} eV/A")
    print(f"selected: mae={selected['mae']:.6f} rmse={selected['rmse']:.6f} max={selected['max']:.6f} eV/A")
    print(f"FORCE_GATE_PASS={ok} (criteria: selected_mae<={args.mae_thresh} and selected_max<={args.max_thresh} eV/A)")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
