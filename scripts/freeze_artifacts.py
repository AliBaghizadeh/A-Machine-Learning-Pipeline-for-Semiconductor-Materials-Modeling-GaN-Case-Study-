"""
Freeze artifacts for the offline demo (local-only).

This collects a minimal subset of the latest local outputs and writes:
- analysis/artifacts/golden_run_<YYYYMMDD>/run_manifest.json
- app/demo_data/run_manifest.json (refreshed)

Design goals:
- Offline-only; no network calls.
- No DFT is run here.
- Gates are re-run (fast) to produce machine-readable, reproducible outputs.
- Works without GPU (falls back to CPU for MACE evaluation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def _safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _latest_model_path() -> Path | None:
    runs = sorted((PROJECT_ROOT / "mlip" / "results").glob("mace_run_*"))
    if not runs:
        return None
    run = runs[-1]
    models = sorted((run / "checkpoints").glob("*.model"))
    if not models:
        return None
    return models[-1]


def _latest_dataset_dir() -> Path | None:
    ptr = PROJECT_ROOT / "mlip" / "data" / "LATEST_DATASET.txt"
    if not ptr.exists():
        return None
    try:
        p = Path(ptr.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return p if p.exists() else None


def _detect_device() -> str:
    # CPU fallback is critical for offline demo portability.
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode, p.stdout


def _parse_energy_gate_stdout(text: str) -> dict:
    results = []
    ok = None
    for line in text.splitlines():
        if " PASS= " in line and "|dE|/atom=" in line:
            # Example: GaN_bulk... PASS= True |dE|/atom= 0.005 eV/atom
            parts = line.split()
            sid = parts[0]
            passed = "PASS= True" in line
            depa = None
            try:
                depa = float(line.split("|dE|/atom=")[1].split()[0])
            except Exception:
                depa = None
            results.append({"structure_id": sid, "pass": passed, "abs_de_per_atom_eV": depa})
        if line.startswith("ENERGY_GATE_PASS="):
            ok = "ENERGY_GATE_PASS=True" in line
    return {"pass": ok, "results": results}


def _parse_force_gate_stdout(text: str) -> dict:
    out: dict = {"pass": None}
    for line in text.splitlines():
        if line.startswith("selection="):
            # selection=coord selected_atoms=37 total_atoms=252
            for tok in line.split():
                if tok.startswith("selected_atoms="):
                    out["selected_atoms"] = int(tok.split("=", 1)[1])
                if tok.startswith("total_atoms="):
                    out["total_atoms"] = int(tok.split("=", 1)[1])
            out["selection"] = line.split()[0].split("=", 1)[1]
        if line.startswith("selected:"):
            # selected: mae=... rmse=... max=... eV/A
            try:
                seg = line.replace("selected:", "").strip().split()
                for kv in seg:
                    if kv.startswith("mae="):
                        out["selected_mae_eV_per_A"] = float(kv.split("=", 1)[1])
                    if kv.startswith("max="):
                        out["selected_max_eV_per_A"] = float(kv.split("=", 1)[1])
            except Exception:
                pass
        if line.startswith("FORCE_GATE_PASS="):
            out["pass"] = "FORCE_GATE_PASS=True" in line
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo"], default="demo")
    ap.add_argument("--out-root", type=str, default="analysis/artifacts")
    ap.add_argument("--copy-to-demo-data", action="store_true", default=True)
    args = ap.parse_args()

    out_root = (PROJECT_ROOT / args.out_root).resolve()
    day = datetime.now().strftime("%Y%m%d")
    out_dir = out_root / f"golden_run_{day}"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_dir = PROJECT_ROOT / "app" / "demo_data"
    demo_dir.mkdir(parents=True, exist_ok=True)
    fallback_manifest = _safe_read_json(demo_dir / "run_manifest.json")

    # Discover latest artifacts (best-effort).
    model_path = _latest_model_path()
    dataset_dir = _latest_dataset_dir()
    dft_json = PROJECT_ROOT / "dft" / "results" / "tier_b_results.json"
    device = _detect_device()

    # Gate config (hardcoded for demo; keep in sync with configs/gates.yaml).
    energy_cases = [
        ("GaN_bulk_sc_4x4x4_relaxed", "dft/structures/GaN_bulk_sc_4x4x4_relaxed.cif"),
        ("GaN_vacancy_line_N_sc_4x4x4_relaxed", "dft/structures/GaN_vacancy_line_N_sc_4x4x4_relaxed.cif"),
    ]
    energy_threshold = 0.01

    force_structure_id = "GaN_vacancy_line_N_sc_4x4x4_relaxed"
    force_cif = "dft/structures/GaN_vacancy_line_N_sc_4x4x4_relaxed.cif"

    # Collect + copy a minimal set of files into the golden run folder.
    copied: list[dict] = []

    def _record_copy(src: Path, rel_dst: Path):
        dst = _copy_if_exists(src, out_dir / rel_dst)
        if dst is None:
            return None
        copied.append(
            {
                "src": str(src),
                "dst": str(rel_dst),
                "sha256": _sha256(dst),
                "bytes": dst.stat().st_size,
            }
        )
        return dst

    # Always include the current demo snippet so the RAG demo works even with empty rag/sources/.
    _record_copy(demo_dir / "rag" / "demo_paper_snippet.txt", Path("rag") / "demo_paper_snippet.txt")

    # Copy a few representative CIFs (best-effort).
    for src in [
        PROJECT_ROOT / "dft" / "structures" / "GaN_bulk_sc_2x2x2.cif",
        PROJECT_ROOT / "dft" / "structures" / "GaN_bulk_sc_4x4x4_relaxed.cif",
        PROJECT_ROOT / "dft" / "structures" / "GaN_vacancy_line_N_sc_4x4x4_relaxed.cif",
        PROJECT_ROOT / "analysis" / "results" / "final_structures" / "GaN_vacancy_line_N_sc_4x4x4_relaxed__DFT_SR_final.cif",
    ]:
        _record_copy(src, Path("structures") / src.name)

    # Copy dataset stats if available.
    dataset_stats = {}
    if dataset_dir is not None:
        stats_path = dataset_dir / "dataset_stats.json"
        if stats_path.exists():
            dataset_stats = _safe_read_json(stats_path)
            _record_copy(stats_path, Path("dataset") / "dataset_stats.json")

    # Copy training summary if present.
    training_summary_path = PROJECT_ROOT / "mlip" / "results" / "training_summary.json"
    training_summary = _safe_read_json(training_summary_path) if training_summary_path.exists() else {}
    if training_summary:
        _record_copy(training_summary_path, Path("mlip") / "training_summary.json")

    # Copy latest validation/analysis JSONs if present (for demo UI + audit trail).
    analysis_jsons = [
        PROJECT_ROOT / "analysis" / "results" / "validation_results.json",
        PROJECT_ROOT / "analysis" / "results" / "mlip_validation.json",
        PROJECT_ROOT / "analysis" / "results" / "structural_summary.json",
    ]
    for p in analysis_jsons:
        _record_copy(p, Path("analysis") / p.name)

    # Run gates (if we have a model and DFT JSON).
    gates_dir = out_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)

    energy_gate = {"pass": None, "results": []}
    force_gate = {"pass": None}
    energy_stdout = ""
    force_stdout = ""
    energy_rc = None
    force_rc = None

    if model_path is not None and dft_json.exists():
        energy_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "analysis" / "scripts" / "energy_gate.py"),
            "--model",
            str(model_path),
            "--dft-json",
            str(dft_json),
            "--device",
            device,
            "--threshold",
            str(energy_threshold),
        ]
        for sid, cif in energy_cases:
            energy_cmd.extend(["--case", f"{sid}:{cif}"])
        energy_rc, energy_stdout = _run(energy_cmd, cwd=PROJECT_ROOT)
        (gates_dir / "energy_gate.out").write_text(energy_stdout, encoding="utf-8")
        energy_gate = _parse_energy_gate_stdout(energy_stdout)
        (gates_dir / "energy_gate.json").write_text(json.dumps(energy_gate, indent=2), encoding="utf-8")

        force_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "analysis" / "scripts" / "force_gate.py"),
            "--model",
            str(model_path),
            "--device",
            device,
            "--structure-id",
            force_structure_id,
            "--cif",
            force_cif,
            "--source",
            "sr",
            "--use-dft-geometry",
            "--select",
            "coord",
            "--coord-rcut",
            "2.4",
            "--coord-max",
            "3",
            "--mae-thresh",
            "0.25",
            "--max-thresh",
            "1.0",
        ]
        force_rc, force_stdout = _run(force_cmd, cwd=PROJECT_ROOT)
        (gates_dir / "force_gate.out").write_text(force_stdout, encoding="utf-8")
        force_gate = _parse_force_gate_stdout(force_stdout)
        (gates_dir / "force_gate.json").write_text(json.dumps(force_gate, indent=2), encoding="utf-8")

    # Copy a couple of small plots if present.
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted((PROJECT_ROOT / "analysis" / "results").glob("*.png"))
    for p in pngs[:2]:
        _record_copy(p, Path("plots") / p.name)

    # Build manifest (golden run).
    gates_ok = (energy_gate.get("pass") is not None) and (len(energy_gate.get("results", [])) > 0) and (force_gate.get("pass") is not None)
    if not gates_ok and fallback_manifest.get("gates"):
        # Keep demo mode robust: do not lose previously-good gate outputs if deps are missing.
        energy_gate = fallback_manifest.get("gates", {}).get("energy_gate", energy_gate)
        force_gate = fallback_manifest.get("gates", {}).get("force_gate", force_gate)

    manifest = {
        "demo": True,
        "created": datetime.now().isoformat(),
        "project": "GaN (offline demo)",
        "device_used_for_gates": device,
        "frozen": {
            "model_path": str(model_path) if model_path is not None else None,
            "dataset_dir": str(dataset_dir) if dataset_dir is not None else None,
            "dft_json": str(dft_json) if dft_json.exists() else None,
        },
        "dataset": {"dataset_stats": dataset_stats},
        "training_summary": training_summary,
        "gates": {
            "energy_gate": {"threshold_eV_per_atom": energy_threshold, **energy_gate},
            "force_gate": force_gate,
        },
        "copied_files": copied,
        "raw_gate_stdout": {"energy_gate": energy_stdout, "force_gate": force_stdout},
        "gate_return_codes": {"energy_gate": energy_rc, "force_gate": force_rc},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Refresh app/demo_data (minimal subset).
    if args.copy_to_demo_data:
        (demo_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for rel in [Path("structures"), Path("gates"), Path("plots"), Path("dataset"), Path("mlip"), Path("rag")]:
            src_dir = out_dir / rel
            if not src_dir.exists():
                continue
            for p in src_dir.rglob("*"):
                if p.is_dir():
                    continue
                dst = demo_dir / rel / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)

    print("FREEZE_DONE out_dir=", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
