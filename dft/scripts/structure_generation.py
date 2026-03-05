"""
Structure Generation for Wurtzite GaN (Minimal-Load)
====================================================

Generate a small, diverse, lightweight set of bulk + point-defect structures
for DFT (GPAW) and MLIP (MACE) training/validation.

This is designed as a pipeline demo:
- Keep cells small (start with 2x2x2 supercells)
- Generate a few off-equilibrium "rattle" variants to get non-zero forces
- Optional point defects: V_Ga and V_N in a small supercell

Outputs:
- CIFs written to: dft/structures/
- Metadata written to: dft/structures/structure_info.json
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from ase.io import read, write
from ase.build import make_supercell


PROJECT_ROOT = Path(__file__).parent.parent.parent
CIFS_DIR = PROJECT_ROOT / "cifs"
STRUCTURES_DIR = PROJECT_ROOT / "dft" / "structures"


DEFAULT_BASE_CIFS = [
    CIFS_DIR / "GaN_mp-804_conventional_standard.cif",
    CIFS_DIR / "GaN.cif",
]

# Keep this minimal for a demo; you can add larger cells later for dislocations.
SUPERCELL_SIZES = [
    (2, 2, 2),  # 32 atoms
]

# Small perturbations for force diversity (MLIP needs forces).
RATTLE_AMP_A = 0.05  # Angstrom (default; override via CLI for gentle rattles)
N_RATTLES_PER_BASE = 2

# Defect configs (point defects); keep minimal.
GENERATE_VACANCIES = True


def _count_atoms(atoms):
    from collections import Counter
    return Counter(a.symbol for a in atoms)


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
    # Small symmetric strain around identity
    eps = (rng.random((3, 3)) - 0.5) * 2.0 * float(amplitude)
    eps = 0.5 * (eps + eps.T)
    new_cell = cell @ (np.eye(3) + eps)
    a.set_cell(new_cell, scale_atoms=True)
    return a


def _create_supercell(atoms, supercell_size):
    nx, ny, nz = map(int, supercell_size)
    transform = np.array([[nx, 0, 0], [0, ny, 0], [0, 0, nz]])
    return make_supercell(atoms, transform)


def _pick_base_cif():
    for p in DEFAULT_BASE_CIFS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No GaN CIF found. Expected one of: "
        + ", ".join(str(p) for p in DEFAULT_BASE_CIFS)
    )


def _central_atom_index(atoms, symbol):
    """Pick the atom of a given symbol closest to the fractional-cell center."""
    idxs = [i for i, a in enumerate(atoms) if a.symbol == symbol]
    if not idxs:
        raise ValueError(f"No atoms with symbol={symbol!r} found.")
    spos = atoms.get_scaled_positions(wrap=True)
    center = np.array([0.5, 0.5, 0.5])
    d2 = []
    for i in idxs:
        v = spos[i] - center
        # account for periodic wrap in fractional coords
        v -= np.round(v)
        d2.append((i, float(np.dot(v, v))))
    d2.sort(key=lambda x: x[1])
    return d2[0][0]


def _make_vacancy(atoms, symbol):
    a = atoms.copy()
    idx = _central_atom_index(a, symbol)
    del a[idx]
    return a, idx


def _write_structure(atoms, struct_id, filename, tags, parent_id=None, notes=None):
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    path = STRUCTURES_DIR / filename
    write(path, atoms)

    counts = _count_atoms(atoms)
    return {
        "id": struct_id,
        "filename": filename,
        "filepath": str(path),
        "n_atoms": int(len(atoms)),
        "atom_counts": dict(counts),
        "supercell_size": getattr(atoms, "info", {}).get("supercell_size"),
        "tags": list(tags),
        "parent_id": parent_id,
        "notes": notes or "",
    }

def _load_structure_info(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_structure_info(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)

def add_rattles(parent_ids, rattle_amp, n_rattles, seed0=91000):
    """
    Append rattled variants for existing structures, without overwriting earlier outputs.

    This is used to add "gentle" rattles after initial bulk/defect set is generated.
    """
    info_file = STRUCTURES_DIR / "structure_info.json"
    info = _load_structure_info(info_file)
    if not info or "structures" not in info:
        raise FileNotFoundError(f"Missing or incompatible {info_file}; run full generation first.")

    by_id = {s.get("id"): s for s in info.get("structures", []) if s.get("id")}
    created = []
    amp_code = int(round(float(rattle_amp) * 100.0))

    for pid in parent_ids:
        meta = by_id.get(pid)
        if meta is None:
            raise ValueError(f"Unknown parent_id={pid!r}. Check dft/structures/structure_info.json")

        base_path = Path(meta["filepath"])
        atoms0 = read(base_path)
        sc = meta.get("supercell_size")
        if sc:
            atoms0.info["supercell_size"] = tuple(sc)

        base_tags = list(meta.get("tags") or [])
        # Ensure rattle tag exists; keep parent's key tags for filtering.
        tags = sorted(set(base_tags + ["rattle"]))

        for j in range(int(n_rattles)):
            rid = f"{pid}__rattle{amp_code:02d}_{j:02d}"
            fname = f"{rid}.cif"
            out_path = STRUCTURES_DIR / fname
            if out_path.exists():
                # Never overwrite; version by timestamp
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                rid = f"{rid}__{ts}"
                fname = f"{rid}.cif"
                out_path = STRUCTURES_DIR / fname

            seed = int(seed0 + 1000 * amp_code + 10 * j)
            rattled = _apply_random_displacements(atoms0, float(rattle_amp), seed=seed)
            rattled.info["supercell_size"] = atoms0.info.get("supercell_size")

            entry = _write_structure(
                rattled,
                struct_id=rid,
                filename=fname,
                tags=tags,
                parent_id=pid,
                notes=f"Added gentle rattle amp={float(rattle_amp)} A seed={seed}",
            )
            info["structures"].append(entry)
            created.append(entry)

    info["last_updated"] = datetime.now().isoformat()
    _save_structure_info(info_file, info)
    print(f"Appended {len(created)} rattles to: {info_file}")
    for e in created:
        print(f"  + {e['id']} ({e['n_atoms']} atoms) tags={e.get('tags')}")
    return created


def generate_all_structures():
    print("\n" + "=" * 60)
    print("STRUCTURE GENERATION FOR GaN (wurtzite)")
    print("=" * 60)

    base_cif = _pick_base_cif()
    print(f"\nBase CIF: {base_cif}")
    base = read(base_cif)

    counts = _count_atoms(base)
    print(f"  Base natoms: {len(base)} counts: {dict(counts)}")
    if counts.get("Ga", 0) == 0 or counts.get("N", 0) == 0:
        raise ValueError("Base CIF does not look like GaN (missing Ga or N).")

    structure_info = {
        "created": datetime.now().isoformat(),
        "system": "GaN",
        "base_cif": str(base_cif),
        "structures": [],
    }

    # Primitive/conventional (both CIFs here are 4-atom wurtzite cells).
    base.info["supercell_size"] = (1, 1, 1)
    structure_info["structures"].append(
        _write_structure(
            base,
            struct_id="GaN_bulk_prim",
            filename="GaN_bulk_prim.cif",
            tags=["bulk", "prim"],
            notes="4-atom wurtzite GaN cell from CIF (P1 setting).",
        )
    )

    # Supercells + optional point defects + rattles.
    for sc in SUPERCELL_SIZES:
        sc_atoms = _create_supercell(base, sc)
        sc_atoms.info["supercell_size"] = tuple(map(int, sc))
        sc_id = f"GaN_bulk_sc_{sc[0]}x{sc[1]}x{sc[2]}"
        sc_file = f"{sc_id}.cif"
        structure_info["structures"].append(
            _write_structure(
                sc_atoms,
                struct_id=sc_id,
                filename=sc_file,
                tags=["bulk", "supercell"],
                parent_id="GaN_bulk_prim",
            )
        )

        bases_to_rattle = [(sc_atoms, sc_id)]

        if GENERATE_VACANCIES:
            vga, idx = _make_vacancy(sc_atoms, "Ga")
            vga.info["supercell_size"] = tuple(map(int, sc))
            vga_id = f"GaN_defect_V_Ga_sc_{sc[0]}x{sc[1]}x{sc[2]}"
            structure_info["structures"].append(
                _write_structure(
                    vga,
                    struct_id=vga_id,
                    filename=f"{vga_id}.cif",
                    tags=["defect", "vacancy", "V_Ga", "supercell"],
                    parent_id=sc_id,
                    notes=f"Removed Ga atom index {idx} (closest to cell center).",
                )
            )

            vn, idx = _make_vacancy(sc_atoms, "N")
            vn.info["supercell_size"] = tuple(map(int, sc))
            vn_id = f"GaN_defect_V_N_sc_{sc[0]}x{sc[1]}x{sc[2]}"
            structure_info["structures"].append(
                _write_structure(
                    vn,
                    struct_id=vn_id,
                    filename=f"{vn_id}.cif",
                    tags=["defect", "vacancy", "V_N", "supercell"],
                    parent_id=sc_id,
                    notes=f"Removed N atom index {idx} (closest to cell center).",
                )
            )

            bases_to_rattle.extend([(vga, vga_id), (vn, vn_id)])

        # Rattle variants (forces diversity) for MLIP.
        for atoms0, parent in bases_to_rattle:
            for j in range(N_RATTLES_PER_BASE):
                seed = 1000 * (sc[0] * 100 + sc[1] * 10 + sc[2]) + 10 * j
                rattled = _apply_random_displacements(atoms0, RATTLE_AMP_A, seed=seed)
                rattled.info["supercell_size"] = tuple(map(int, sc))
                rid = f"{parent}__rattle{j:02d}"
                structure_info["structures"].append(
                    _write_structure(
                        rattled,
                        struct_id=rid,
                        filename=f"{rid}.cif",
                        tags=["rattle"] + [t for t in ("bulk" if "bulk" in parent else "defect",)],
                        parent_id=parent,
                        notes=f"Random displacement rattle amp={RATTLE_AMP_A} A seed={seed}",
                    )
                )

        # One tiny strain variant for bulk supercell only.
        seed = 2000 * (sc[0] * 100 + sc[1] * 10 + sc[2])
        strained = _apply_random_strain(sc_atoms, amplitude=0.01, seed=seed)
        strained.info["supercell_size"] = tuple(map(int, sc))
        sid = f"{sc_id}__strain00"
        structure_info["structures"].append(
            _write_structure(
                strained,
                struct_id=sid,
                filename=f"{sid}.cif",
                tags=["bulk", "supercell", "strain"],
                parent_id=sc_id,
                notes="Small random symmetric strain (1%) for off-equilibrium data.",
            )
        )

    info_file = STRUCTURES_DIR / "structure_info.json"
    _save_structure_info(info_file, structure_info)

    print("\n" + "=" * 60)
    print("STRUCTURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total structures: {len(structure_info['structures'])}")
    print(f"Structures dir: {STRUCTURES_DIR}")
    print(f"Metadata: {info_file}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate GaN structures for DFT/MLIP")
    p.add_argument("--rattle-amp", type=float, default=RATTLE_AMP_A, help="Rattle amplitude in Angstrom (default: 0.05)")
    p.add_argument("--n-rattles-per-base", type=int, default=N_RATTLES_PER_BASE, help="Rattles per base structure (default: 2)")
    p.add_argument("--add-rattles-for", nargs="+", default=None, help="Append rattles for existing structure IDs (no overwrite).")
    args = p.parse_args()

    # Allow overriding defaults for full generation mode
    if args.add_rattles_for:
        add_rattles(args.add_rattles_for, args.rattle_amp, args.n_rattles_per_base)
    else:
        # Update module constants for this run
        RATTLE_AMP_A = float(args.rattle_amp)
        N_RATTLES_PER_BASE = int(args.n_rattles_per_base)
        generate_all_structures()
