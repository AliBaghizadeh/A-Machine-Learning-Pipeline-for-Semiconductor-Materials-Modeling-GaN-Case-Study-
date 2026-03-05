"""Narrow offline RAG report generator.

Output intent (showcase-friendly):
- extracted parameters + paper title table
- concise report markdown without noisy chunk-level details
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_golden_run(out_root: Path) -> Path | None:
    runs = sorted(out_root.glob("golden_run_*/run_manifest.json"))
    if not runs:
        return None
    return runs[-1].parent


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _paper_title_from_row(r: Dict[str, str]) -> str:
    if r.get("paper_title"):
        return str(r["paper_title"])
    cit = str(r.get("citation") or "")
    return cit.split("#")[0].strip() if cit else "unknown_source"


def _clean_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for r in rows:
        param = str(r.get("param") or "").strip()
        value = str(r.get("value") or "").strip()
        paper_title = _paper_title_from_row(r)
        if not param or not value or not paper_title:
            continue
        key = (param, value, paper_title)
        if key in seen:
            continue
        seen.add(key)
        out.append({"param": param, "value": value, "paper_title": paper_title})
    out.sort(key=lambda x: (x["param"], x["paper_title"], x["value"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True)
    ap.add_argument("--demo-data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    golden = _find_latest_golden_run(out_root)
    if golden is None:
        golden = out_root / ("rag_report_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        golden.mkdir(parents=True, exist_ok=True)

    rag_dir = golden / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)

    index_dir = Path(args.index)
    table = index_dir / "gan_dft_params_table.csv"
    if not table.exists():
        raise SystemExit(f"Missing param table: {table} (run make rag-ingest)")

    rows_raw = _read_csv_rows(table)
    rows = _clean_rows(rows_raw)

    manifest = _load_json(Path(args.demo_data) / "run_manifest.json")
    gates = manifest.get("gates") or {}
    eg = gates.get("energy_gate") or {}
    fg = gates.get("force_gate") or {}

    # Write clean table for app consumption.
    clean_csv = rag_dir / "extracted_params_by_paper.csv"
    _write_csv(clean_csv, rows, fieldnames=["param", "value", "paper_title"])

    # Keep original table copy for traceability.
    (rag_dir / "gan_dft_params_table.csv").write_text(table.read_text(encoding="utf-8"), encoding="utf-8")

    report = rag_dir / "rag_report.md"
    lines: List[str] = []
    lines.append("# RAG Report: GaN DFT Parameters\n")
    lines.append(f"Generated: `{datetime.now().isoformat()}`\n")
    lines.append("\n## What This Shows\n")
    lines.append("- Extracted GaN DFT parameters from local papers (offline).\n")
    lines.append("- Focused table: parameter, value, paper title.\n")

    lines.append("\n## Extracted Parameters By Paper\n")
    if rows:
        lines.append("| param | value | paper_title |\n")
        lines.append("|---|---|---|\n")
        for r in rows[:200]:
            lines.append(f"| {r['param']} | {r['value']} | {r['paper_title']} |\n")
    else:
        lines.append("_No parameters extracted from current sources._\n")

    lines.append("\n## Local Run Gate Snapshot\n")
    lines.append(f"- Energy gate pass: `{eg.get('pass')}`\n")
    lines.append(f"- Force gate pass: `{fg.get('pass')}`\n")

    report.write_text("".join(lines), encoding="utf-8")

    print("RAG_REPORT_DONE out=", rag_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
