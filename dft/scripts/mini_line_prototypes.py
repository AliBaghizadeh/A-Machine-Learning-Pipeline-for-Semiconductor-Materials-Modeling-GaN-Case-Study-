#!/usr/bin/env python3
"""
Mini Line-Prototypes for GaN (Minimal-Load)
==========================================

Create small supercells that mimic "line-ish" under-coordination environments
without using large 252/576-atom cells. These are intended for quick Tier-B DFT
labels to fix MLIP extrapolation on vacancy-line structures.

Outputs:
- CIFs written to: dft/structures/
- Pointer JSON written to: dft/structures/mini_line_latest.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.build import make_supercell


PROJECT_ROOT = Path(__file__).parent.parent.parent
CIFS_DIR = PROJECT_ROOT / "cifs"
STRUCTURES_DIR = PROJECT_ROOT / "dft" / "structures"


def _create_supercell(atoms, supercell_size):
    nx, ny, nz = map(int, supercell_size)
    transform = np.array([[nx, 0, 0], [0, ny, 0], [0, 0, nz]])
    sc = make_supercell(atoms, transform)
    sc.info["supercell_size"] = (nx, ny, nz)
    return sc


def _apply_random_displacements(atoms, amplitude, seed):
    rng = np.random.default_rng(int(seed))
    a = atoms.copy()
    disp = rng.normal(size=(len(a), 3)) * float(amplitude)
    a.set_positions(a.get_positions() + disp)
    return a


def _apply_random_strain(atoms, amplitude, seed):
    rng = np.random.default_rng(int(seed))
    a = atoms.copy()
    cell = a.get_cell()
    eps = (rng.random((3, 3)) - 0.5) * 2.0 * float(amplitude)
    eps = 0.5 * (eps + eps.T)
    new_cell = cell @ (np.eye(3) + eps)
    a.set_cell(new_cell, scale_atoms=True)
    return a


def _xy_distance_frac(a, b):
    """Periodic distance in fractional xy (wrap-aware)."""
    d = a[:2] - b[:2]
    d -= np.round(d)
    return float(np.linalg.norm(d))


def _pick_column_indices(atoms, symbol: str, n_remove: int, eps_xy: float = 0.06):
    """
    Pick indices of atoms to remove to form a short 'column' along c.
    Strategy:
    - Choose a seed atom of the given symbol closest to fractional center (0.5,0.5) in xy.
    - Find atoms of same symbol with nearly same xy (within eps_xy).
    - Remove up to n_remove atoms closest to z=0.5 (gives a short segment).
    """
    symbols = atoms.get_chemical_symbols()
    idxs = [i for i, s in enumerate(symbols) if s == symbol]
    if not idxs:
        raise ValueError(f"No atoms with symbol={symbol!r} found.")

    spos = atoms.get_scaled_positions(wrap=True)
    center_xy = np.array([0.5, 0.5])
    # Pick seed closest to center in xy.
    best = min(idxs, key=lambda i: _xy_distance_frac(spos[i], np.array([center_xy[0], center_xy[1], spos[i][2]])))
    seed_xy = spos[best].copy()

    # Same xy column (within eps).
    col = [i for i in idxs if _xy_distance_frac(spos[i], seed_xy) <= float(eps_xy)]
    if len(col) < n_remove:
        # If too few, just remove the n closest in xy to the seed.
        col = sorted(idxs, key=lambda i: _xy_distance_frac(spos[i], seed_xy))[:n_remove]

    # Prefer atoms closest to mid-plane in z.
    col = sorted(col, key=lambda i: abs(float(spos[i][2]) - 0.5))
    return col[: int(n_remove)]


def _write_cif(atoms, struct_id: str):
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    path = STRUCTURES_DIR / f"{struct_id}.cif"
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        struct_id = f"{struct_id}__{ts}"
        path = STRUCTURES_DIR / f"{struct_id}.cif"
    write(path, atoms)
    return struct_id, path


def main() -> int:
    p = argparse.ArgumentParser(description="Generate small GaN mini line-prototypes for MLIP training")
    p.add_argument("--base-cif", type=str, default=str(CIFS_DIR / "GaN_mp-804_conventional_standard.cif"))
    p.add_argument("--supercell", nargs=3, type=int, default=[3, 3, 2], help="Supercell size (default: 3 3 2)")
    p.add_argument("--n-remove", type=int, default=2, help="Number of N atoms to remove along the column (default: 2)")
    p.add_argument("--rattle-amp", type=float, default=0.02, help="Rattle amplitude in Angstrom (default: 0.02)")
    p.add_argument("--n-rattles", type=int, default=2, help="Number of rattles for mini-line prototype (default: 2)")
    p.add_argument("--strain-amp", type=float, default=0.01, help="Random symmetric strain amplitude (default: 0.01)")
    args = p.parse_args()

    base = read(Path(args.base_cif))
    sc = tuple(int(x) for x in args.supercell)

    bulk = _create_supercell(base, sc)
    bulk_id = f"GaN_bulk_sc_{sc[0]}x{sc[1]}x{sc[2]}"
    bulk_id, bulk_path = _write_cif(bulk, bulk_id)

    # Mini line: remove a short N column segment.
    line = bulk.copy()
    idx_remove = _pick_column_indices(line, symbol="N", n_remove=int(args.n_remove))
    # Delete in descending order so indices remain valid.
    for idx in sorted(idx_remove, reverse=True):
        del line[idx]
    line.info["supercell_size"] = bulk.info.get("supercell_size")

    line_id = f"GaN_defect_line_V_N_sc_{sc[0]}x{sc[1]}x{sc[2]}"
    line_id, line_path = _write_cif(line, line_id)

    created = {
        "created": datetime.now().isoformat(),
        "supercell": sc,
        "bulk": {"id": bulk_id, "cif": str(bulk_path), "natoms": int(len(bulk))},
        "line": {
            "id": line_id,
            "cif": str(line_path),
            "natoms": int(len(line)),
            "removed_indices_in_bulk": list(map(int, idx_remove)),
            "n_removed": int(len(idx_remove)),
        },
        "variants": {},
    }

    # Rattles (mini-line only).
    rattles = []
    amp_code = int(round(float(args.rattle_amp) * 100.0))
    for j in range(int(args.n_rattles)):
        seed = 91000 + 1000 * amp_code + 10 * j
        r = _apply_random_displacements(line, float(args.rattle_amp), seed=seed)
        r.info["supercell_size"] = line.info.get("supercell_size")
        rid = f"{line_id}__rattle{amp_code:02d}_{j:02d}"
        rid, rpath = _write_cif(r, rid)
        rattles.append({"id": rid, "cif": str(rpath), "natoms": int(len(r)), "seed": int(seed)})
    created["variants"]["rattles"] = rattles

    # Strain (mini-line only).
    seed = 2000 * (sc[0] * 100 + sc[1] * 10 + sc[2])
    strained = _apply_random_strain(line, float(args.strain_amp), seed=seed)
    strained.info["supercell_size"] = line.info.get("supercell_size")
    sid = f"{line_id}__strain00"
    sid, spath = _write_cif(strained, sid)
    created["variants"]["strain"] = {"id": sid, "cif": str(spath), "natoms": int(len(strained)), "seed": int(seed)}

    out_ptr = STRUCTURES_DIR / "mini_line_latest.json"
    with open(out_ptr, "w", encoding="utf-8") as f:
        json.dump(created, f, indent=2)

    print("Created mini line-prototypes:")
    print(f"  bulk: {created['bulk']['id']} ({created['bulk']['natoms']} atoms) -> {created['bulk']['cif']}")
    print(f"  line: {created['line']['id']} ({created['line']['natoms']} atoms) -> {created['line']['cif']}")
    for rr in created["variants"]["rattles"]:
        print(f"  rattle: {rr['id']} ({rr['natoms']} atoms) -> {rr['cif']}")
    print(f"  strain: {created['variants']['strain']['id']} ({created['variants']['strain']['natoms']} atoms) -> {created['variants']['strain']['cif']}")
    print(f"Pointer written: {out_ptr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

