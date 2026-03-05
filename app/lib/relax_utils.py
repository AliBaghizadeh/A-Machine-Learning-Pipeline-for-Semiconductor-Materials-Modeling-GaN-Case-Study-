from __future__ import annotations

from pathlib import Path
from typing import Any


def summarize_atoms_state(atoms: Any) -> dict:
    """
    Return lightweight relaxation metrics for the current atomic state.

    The function is defensive: if energy or forces fail, fields stay None.
    """
    out = {"energy_eV": None, "max_force_eV_per_A": None}
    try:
        out["energy_eV"] = float(atoms.get_potential_energy())
    except Exception:
        out["energy_eV"] = None
    try:
        forces = atoms.get_forces()
        if forces is not None and len(forces) > 0:
            norms = (forces**2).sum(axis=1) ** 0.5
            out["max_force_eV_per_A"] = float(norms.max())
    except Exception:
        out["max_force_eV_per_A"] = None
    return out


def read_text_preview(path: str | Path, *, max_lines: int = 40) -> str:
    """Read first max_lines from a text file; return friendly fallback on errors."""
    p = Path(path)
    if not p.exists():
        return "Log file not found."
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[:max_lines]) if lines else "(empty log)"
    except Exception as e:
        return f"Could not read log preview: {e!r}"


def format_float(v: Any, *, digits: int = 6, fallback: str = "unknown") -> str:
    """Format numeric values consistently for UI metrics."""
    try:
        if v is None:
            return fallback
        return f"{float(v):.{digits}f}"
    except Exception:
        return fallback
