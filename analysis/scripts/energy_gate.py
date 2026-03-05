#!/usr/bin/env python3
"""
Energy gate: compare MLIP energy vs DFT Tier-B single-point energy for specific structures.

Why this exists:
- We previously used inline one-off `python - <<PY` blocks for energy gates.
- For audit/reproducibility, we want a single script that defines:
  - how DFT reference energies are picked
  - how |dE|/atom is computed
  - how structure_id maps to a CIF path
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_case(s: str) -> Tuple[str, str]:
    # Accept "sid:path" (path may contain ':' on Windows, but we are on Linux here).
    if ":" not in s:
        raise argparse.ArgumentTypeError("case must be 'STRUCTURE_ID:PATH_TO_CIF'")
    sid, path = s.split(":", 1)
    sid = sid.strip()
    path = path.strip()
    if not sid or not path:
        raise argparse.ArgumentTypeError("case must be 'STRUCTURE_ID:PATH_TO_CIF'")
    return sid, path


def _load_dft_sp_row(dft: Dict[str, Any], structure_id: str) -> Dict[str, Any]:
    rows = [
        x
        for x in dft.get("single_point", [])
        if x.get("status") == "completed" and x.get("structure_id") == structure_id
    ]
    if not rows:
        raise KeyError(f"No completed DFT single_point entry found for structure_id={structure_id}")
    return rows[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Energy gate: MLIP vs DFT single-point |dE|/atom")
    ap.add_argument(
        "--model",
        required=True,
        help="Path to MACE deployable model (*.model).",
    )
    ap.add_argument(
        "--dft-json",
        default="dft/results/tier_b_results.json",
        help="Path to DFT Tier-B results JSON (default: dft/results/tier_b_results.json).",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="MACE device (default: cuda).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Acceptance threshold for |dE|/atom in eV/atom (default: 0.01).",
    )
    ap.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        default=[],
        help="Repeatable: STRUCTURE_ID:PATH_TO_CIF",
    )
    args = ap.parse_args()

    model_path = Path(args.model)
    dft_json = Path(args.dft_json)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not dft_json.exists():
        raise FileNotFoundError(dft_json)
    if not args.case:
        raise SystemExit("No --case provided. Example: --case GaN_bulk_sc_4x4x4_relaxed:dft/structures/GaN_bulk_sc_4x4x4_relaxed.cif")

    # Lazy imports: keep CLI snappy when only validating args.
    from ase.io import read
    from mace.calculators import MACECalculator

    dft = json.load(open(dft_json))
    calc = MACECalculator(model_paths=str(model_path), device=str(args.device))

    print("energy_gate:")
    print(f"  model={model_path}")
    print(f"  dft_json={dft_json}")
    print(f"  device={args.device}")
    print(f"  threshold={args.threshold} eV/atom")
    print("  cases:")
    for sid, cif in args.case:
        print(f"    - {sid} -> {cif}")

    ok_all = True
    for sid, cif in args.case:
        cif_path = Path(cif)
        if not cif_path.exists():
            raise FileNotFoundError(cif_path)

        a = read(str(cif_path))
        a.calc = calc
        e_mlip = float(a.get_potential_energy())

        row = _load_dft_sp_row(dft, sid)
        e_dft = float(row["energy"])
        n = max(int(row.get("n_atoms") or len(row.get("forces", [])) or len(a)), 1)
        depa = abs((e_mlip - e_dft) / n)
        k = depa <= float(args.threshold)
        ok_all = ok_all and k
        print(f"{sid} PASS= {k} |dE|/atom= {depa} eV/atom")

    print(f"ENERGY_GATE_PASS={ok_all} (criterion: all |dE|/atom <= {args.threshold} eV/atom)")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())

