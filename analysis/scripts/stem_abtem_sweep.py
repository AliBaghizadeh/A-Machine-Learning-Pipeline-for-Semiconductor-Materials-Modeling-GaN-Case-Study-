"""Small parameter sweep for abTEM HAADF to pick practical demo settings.

Runs a bounded grid over defocus and detector angles, computes quick image
quality proxies, and ranks candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _parse_list_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _metrics(arr: np.ndarray) -> dict:
    arr = arr.astype(np.float64)
    p1, p99 = np.percentile(arr, [1, 99])
    contrast = float(p99 - p1)
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    edge = float(np.mean(np.abs(gx) + np.abs(gy)))
    score = 0.5 * contrast + 0.5 * edge
    return {"contrast_p99_p1": contrast, "edge_mean_abs_grad": edge, "score": score}


def main() -> int:
    ap = argparse.ArgumentParser(description="Bounded abTEM HAADF parameter sweep")
    ap.add_argument("--structure", required=True)
    ap.add_argument("--out-dir", default="analysis/results/stem_sweep")
    ap.add_argument(
        "--defocus-list",
        default="-10,0,10,30",
        help="Comma-separated defocus values in Angstrom. Use quotes or '=' when first value is negative, e.g. --defocus-list='-10,0,10,30'",
    )
    ap.add_argument("--inner-list", default="18,24")
    ap.add_argument("--outer-list", default="30,36")
    ap.add_argument("--semiangle-cutoff", type=float, default=36.0)
    ap.add_argument("--energy-kv", type=float, default=300.0)
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--scan-divisor", type=float, default=4.0, help="Higher = faster")
    ap.add_argument("--repeat", default="1,1,1")
    ap.add_argument("--max-cases", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    defocus_list = _parse_list_floats(args.defocus_list)
    inner_list = _parse_list_floats(args.inner_list)
    outer_list = _parse_list_floats(args.outer_list)

    rows = []
    case_idx = 0
    for d in defocus_list:
        for inner in inner_list:
            for outer in outer_list:
                if outer <= inner:
                    continue
                if outer > args.semiangle_cutoff:
                    continue
                case_idx += 1
                if case_idx > args.max_cases:
                    break

                tag = f"case_{case_idx:02d}_df{d:+.0f}_in{inner:.0f}_out{outer:.0f}".replace("+", "p").replace("-", "m")
                cmd = [
                    sys.executable,
                    "analysis/scripts/stem_abtem_haadf.py",
                    "--structure",
                    args.structure,
                    "--out-dir",
                    str(out_dir),
                    "--repeat",
                    args.repeat,
                    "--energy-kv",
                    str(args.energy_kv),
                    "--device",
                    str(args.device),
                    "--semiangle-cutoff",
                    str(args.semiangle_cutoff),
                    "--defocus-a",
                    str(d),
                    "--haadf-inner",
                    str(inner),
                    "--haadf-outer",
                    str(outer),
                    "--scan-divisor",
                    str(args.scan_divisor),
                    "--tag",
                    tag,
                ]
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if p.returncode != 0:
                    rows.append(
                        {
                            "tag": tag,
                            "defocus_A": d,
                            "inner_mrad": inner,
                            "outer_mrad": outer,
                            "status": "failed",
                            "device": args.device,
                            "score": "",
                            "contrast_p99_p1": "",
                            "edge_mean_abs_grad": "",
                            "log": p.stdout[:4000],
                        }
                    )
                    continue

                arr = np.load(out_dir / f"{tag}.npy")
                m = _metrics(arr)
                rows.append(
                    {
                        "tag": tag,
                        "defocus_A": d,
                        "inner_mrad": inner,
                        "outer_mrad": outer,
                            "status": "ok",
                            "device": args.device,
                            "score": m["score"],
                            "contrast_p99_p1": m["contrast_p99_p1"],
                            "edge_mean_abs_grad": m["edge_mean_abs_grad"],
                        "log": "",
                    }
                )
            if case_idx > args.max_cases:
                break
        if case_idx > args.max_cases:
            break

    ok_rows = [r for r in rows if r["status"] == "ok"]
    ok_rows.sort(key=lambda r: float(r["score"]), reverse=True)

    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tag",
            "defocus_A",
            "inner_mrad",
            "outer_mrad",
            "status",
            "device",
            "score",
            "contrast_p99_p1",
            "edge_mean_abs_grad",
            "log",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "cases_total": len(rows),
        "cases_ok": len(ok_rows),
        "best": ok_rows[0] if ok_rows else None,
        "csv": str(csv_path),
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("STEM_SWEEP_DONE")
    print("  out_dir=", out_dir)
    print("  csv=", csv_path)
    if ok_rows:
        b = ok_rows[0]
        print("  best=", b["tag"], "score=", b["score"], "defocus=", b["defocus_A"], "inner=", b["inner_mrad"], "outer=", b["outer_mrad"])
    else:
        print("  best= none (all failed)")
        first_err = next((r for r in rows if r["status"] == "failed" and r.get("log")), None)
        if first_err:
            print("  first_error_hint:")
            print((first_err.get("log") or "")[:600].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
