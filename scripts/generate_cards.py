"""
Generate Model Card + Dataset Card from a frozen run manifest.

This is demo-focused and intentionally lightweight:
- Prefer latest analysis/artifacts/golden_run_*/run_manifest.json
- Fall back to app/demo_data/run_manifest.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_golden_manifest() -> Path | None:
    root = PROJECT_ROOT / "analysis" / "artifacts"
    if not root.exists():
        return None
    candidates = sorted(root.glob("golden_run_*/run_manifest.json"))
    return candidates[-1] if candidates else None


def _get(d: dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> int:
    manifest_path = _latest_golden_manifest() or (PROJECT_ROOT / "app" / "demo_data" / "run_manifest.json")
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    m = _load_json(manifest_path)
    now = datetime.now().isoformat()

    model_path = _get(m, "frozen.model_path")
    dataset_dir = _get(m, "frozen.dataset_dir")
    dft_json = _get(m, "frozen.dft_json")

    eg = _get(m, "gates.energy_gate", {}) or {}
    fg = _get(m, "gates.force_gate", {}) or {}
    stats = _get(m, "dataset.dataset_stats", {}) or {}

    model_card = (
        "# Model Card: GaN MACE MLIP (Demo)\n\n"
        f"Generated: `{now}`\n\n"
        "## Overview\n"
        "This model card describes the *demo* MACE MLIP artifacts used in this repository.\n"
        "It is intended to showcase an offline, production-style workflow (not publication-grade physics).\n\n"
        "## Frozen Artifacts\n"
        f"- Frozen manifest: `{manifest_path}`\n"
        f"- Model path: `{model_path}`\n"
        f"- Dataset dir: `{dataset_dir}`\n"
        f"- DFT reference JSON: `{dft_json}`\n\n"
        "## Intended Use\n"
        "- Offline demo of an MLIP pipeline: dataset extraction -> MACE training -> validation gates -> iteration.\n"
        "- Optional local MLIP relaxations for quick geometry exploration (no DFT from the app).\n\n"
        "## Metrics (Gates)\n"
        f"- Energy gate pass: `{eg.get('pass')}` (threshold `{eg.get('threshold_eV_per_atom')}` eV/atom)\n"
        f"- Energy gate cases: `{len(eg.get('results', []) or [])}`\n"
        f"- Force gate pass: `{fg.get('pass')}`\n"
        f"- Force gate selection: `{fg.get('selection')}` (selected `{fg.get('selected_atoms')}` / total `{fg.get('total_atoms')}`)\n\n"
        "## Limitations\n"
        "- Demo-only dataset sizes are small and correlated.\n"
        "- Gate pass/fail is a narrow correctness check, not a universal accuracy guarantee.\n"
        "- Results depend on local GPAW settings and the specific Tier-B reference calculations stored in the repo.\n\n"
        "## Reproducibility\n"
        "- Run `make demo-artifacts` to regenerate the manifest + cards from current local outputs.\n"
    )

    dataset_card = (
        "# Dataset Card: GaN DFT Labels for MLIP (Demo)\n\n"
        f"Generated: `{now}`\n\n"
        "## Overview\n"
        "This dataset card describes the *demo* DFT-derived labels used to train the MACE model.\n\n"
        "## Frozen Artifacts\n"
        f"- Frozen manifest: `{manifest_path}`\n"
        f"- Dataset dir: `{dataset_dir}`\n\n"
        "## Composition / Size (from dataset_stats)\n"
        f"- total_structures: `{stats.get('total_structures')}`\n"
        f"- total_structures_full: `{stats.get('total_structures_full')}`\n"
        f"- max_atoms_filter: `{stats.get('max_atoms_filter')}`\n"
        f"- train/val/test: `{stats.get('train')}` / `{stats.get('val')}` / `{stats.get('test')}`\n"
        f"- sources: `{stats.get('sources')}`\n\n"
        "## Label Types\n"
        "- Energies (eV)\n"
        "- Forces (eV/Angstrom)\n"
        "- Stress (when available)\n\n"
        "## Known Limitations\n"
        "- Small, demo-scale dataset; not meant for general-purpose deployment.\n"
        "- Coverage is biased toward the structures used in the gate/iteration loop.\n\n"
        "## Reproducibility\n"
        "- Dataset is produced by `dft/scripts/extract_dft_data.py` and tracked by `mlip/data/LATEST_DATASET.txt`.\n"
    )

    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "model_card.md").write_text(model_card, encoding="utf-8")
    (docs_dir / "dataset_card.md").write_text(dataset_card, encoding="utf-8")

    print("CARDS_DONE manifest=", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

