#!/usr/bin/env python3
"""
Run a very short DFT MD (BO-MD) using GPAW via ASE and export a few frames to CIF.

Design goals:
- Minimal compute (few steps) to de-correlate the training set with realistic thermal distortions.
- Reproducible: fixed seed, explicit exported frames, and a pointer JSON written to dft/structures/.
- Reuse Tier-B GPAW settings (kpts scaling, GPU PW mode) for consistency.

Typical use (5 steps, 300 K, export 3 frames):
  python dft/scripts/dft_md_snapshots.py --cif dft/structures/GaN_bulk_sc_2x2x2.cif --gpu \\
    --temperature 300 --steps 5 --dt-fs 1.0 --export-frames 0,2,5
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
from ase import units


def _run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_frames(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("export frames list is empty")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Short DFT MD and snapshot export (GPAW+ASE)")
    ap.add_argument("--cif", required=True, help="Input CIF path (initial geometry).")
    ap.add_argument("--gpu", action="store_true", help="Use GPAW GPU mode (PW+CuPy).")
    ap.add_argument("--temperature", type=float, default=300.0, help="Temperature in K (default: 300).")
    ap.add_argument("--steps", type=int, default=5, help="MD steps (default: 5).")
    ap.add_argument("--dt-fs", type=float, default=1.0, help="Timestep in fs (default: 1.0).")
    ap.add_argument("--friction", type=float, default=0.02, help="Langevin friction in 1/fs (default: 0.02).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for velocities (default: 0).")
    ap.add_argument("--export-frames", default="0,2,5", help="Comma list of trajectory indices to export (default: 0,2,5).")
    ap.add_argument("--out-prefix", default=None, help="Prefix for exported CIF structure IDs (default: derived from CIF stem).")
    ap.add_argument("--out-structures-dir", default="dft/structures", help="Where to write exported CIFs.")
    ap.add_argument("--out-traj", default=None, help="Trajectory output path (default: dft/results/trajectories/dft_md_<stem>__<tag>.traj).")
    ap.add_argument("--pointer-json", default="dft/structures/dft_md_latest.json", help="Pointer JSON to write (default: dft/structures/dft_md_latest.json).")
    args = ap.parse_args()

    cif_path = Path(args.cif)
    if not cif_path.exists():
        raise FileNotFoundError(cif_path)

    # Import here so the script is still "importable" without GPAW, but fails at runtime use.
    # Reuse Tier-B settings for consistency (kpts scaling, GPU mode, etc.).
    import sys

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from dft.scripts.tier_b_calculations import create_gpaw_calculator_tierb, _infer_supercell_size  # type: ignore

    out_structures_dir = Path(args.out_structures_dir)
    out_structures_dir.mkdir(parents=True, exist_ok=True)

    tag = _run_tag()
    stem = cif_path.stem
    prefix = args.out_prefix or f"{stem}__md{int(round(args.temperature))}K"

    if args.out_traj:
        traj_path = Path(args.out_traj)
    else:
        traj_path = project_root / "dft" / "results" / "trajectories" / f"dft_md_{stem}__{tag}.traj"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    pointer_path = Path(args.pointer_json)
    if not pointer_path.is_absolute():
        pointer_path = project_root / pointer_path

    export_frames = _parse_frames(args.export_frames)

    atoms = read(str(cif_path))
    supercell_size = _infer_supercell_size(atoms)

    # GPAW calculator (Tier-B-like). MD calls forces repeatedly.
    try:
        calc = create_gpaw_calculator_tierb(
            calc_type="single_point",
            supercell_size=tuple(supercell_size),
            gpu=bool(args.gpu),
            scf_overrides=None,
            mag_config="none",
            txt_path=None,
            gpaw_overrides={"symmetry": "off"},
        )
    except TypeError:
        calc = create_gpaw_calculator_tierb(
            calc_type="single_point",
            supercell_size=tuple(supercell_size),
            gpu=bool(args.gpu),
            scf_overrides=None,
            mag_config="none",
            txt_path=None,
            gpaw_overrides={"symmetry": {"point_group": False, "time_reversal": True}},
        )
    atoms.calc = calc

    # Initialize velocities (reproducible)
    rng = np.random.default_rng(int(args.seed))
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(args.temperature), rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    dt = float(args.dt_fs) * units.fs
    dyn = Langevin(atoms, dt, temperature_K=float(args.temperature), friction=float(args.friction))

    # IMPORTANT: Ensure GPAW has produced results before the first Trajectory write.
    # Trajectory.write() will query calculator properties (energy/forces/...) and
    # GPAW (new ASE interface) can raise AttributeError if no calculation has run yet.
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()

    # Write trajectory: include initial frame at index 0.
    traj = Trajectory(str(traj_path), "w", atoms)
    traj.write(atoms)
    dyn.attach(traj.write, interval=1)

    t0 = time.time()
    dyn.run(int(args.steps))
    elapsed = time.time() - t0
    traj.close()

    # Export selected frames to CIF
    exported = []
    # Read all frames via ASE; safer than keeping Trajectory handle open.
    for idx in export_frames:
        a = read(str(traj_path), index=idx)
        sid = f"{prefix}_frame{idx:03d}__{tag}"
        out_cif = out_structures_dir / f"{sid}.cif"
        write(str(out_cif), a)
        exported.append(
            {
                "structure_id": sid,
                "frame_index": idx,
                "cif": str(out_cif.resolve()),
                "natoms": len(a),
            }
        )

    payload = {
        "run_tag": tag,
        "input_cif": str(cif_path.resolve()),
        "trajectory": str(traj_path.resolve()),
        "temperature_K": float(args.temperature),
        "steps": int(args.steps),
        "dt_fs": float(args.dt_fs),
        "friction_1_per_fs": float(args.friction),
        "seed": int(args.seed),
        "gpu": bool(args.gpu),
        "supercell_size": list(map(int, supercell_size)),
        "exported_frames": exported,
        "elapsed_s": float(elapsed),
    }
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pointer_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("DFT_MD_DONE")
    print("  input_cif=", payload["input_cif"])
    print("  traj=", payload["trajectory"])
    print("  exported_n=", len(exported))
    for x in exported:
        print("   ", x["structure_id"], "->", x["cif"])
    print("  pointer=", str(pointer_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
