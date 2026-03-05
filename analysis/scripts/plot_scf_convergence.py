"""
Plot GPAW SCF convergence from a text log file.

Usage example:
python analysis/scripts/plot_scf_convergence.py \
  --log dft/results/logs/gpaw_tierb_sp.out \
  --out analysis/results/scf_convergence_tierb_sp.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_float_token(token: str):
    m = FLOAT_RE.search(token)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_gpaw_scf_iterations(log_path: Path):
    """
    Parse lines like:
    iter:   2 23:52:42 -1013.019771 +0.61 -0.94 +109.4724
    """
    rows = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("iter:"):
                continue
            tokens = line.split()
            if len(tokens) < 6:
                continue

            # Base columns (always expected):
            # 0='iter:' 1=iter 2=time 3=energy ... last=magmom
            iter_i = parse_float_token(tokens[1])
            energy = parse_float_token(tokens[3])
            magmom = parse_float_token(tokens[-1])

            if iter_i is None or energy is None:
                continue

            # Middle convergence tokens can be:
            # [eigst] / [eigst, dens] / [eigst, dens, force]
            middle = tokens[4:-1]
            eigst = parse_float_token(middle[0]) if len(middle) >= 1 else None
            dens = parse_float_token(middle[1]) if len(middle) >= 2 else None
            force = parse_float_token(middle[2]) if len(middle) >= 3 else None

            rows.append(
                {
                    "iter": int(iter_i),
                    "energy": energy,
                    "eigst": eigst,
                    "dens": dens,
                    "force": force,
                    "magmom": magmom,
                }
            )

    return rows


def write_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "energy", "eigst", "dens", "force", "magmom"])
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, out_path: Path, title: str):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        if "matplotlib" in str(e):
            raise
        raise

    out_path.parent.mkdir(parents=True, exist_ok=True)

    it = [r["iter"] for r in rows]

    series = [
        ("energy", "Total Energy (eV)"),
        ("eigst", "log10-change eigst"),
        ("dens", "log10-change dens"),
        ("force", "log10-change force"),
        ("magmom", "Magnetic Moment"),
    ]

    available = [s for s in series if any(r[s[0]] is not None for r in rows)]
    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.2 * n, 5)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (key, ylabel) in zip(axes, available):
        y = [r[key] for r in rows]
        ax.plot(it, y, marker="o", markersize=3, linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("SCF Iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GPAW SCF convergence from log file")
    parser.add_argument("--log", required=True, help="Path to GPAW text log (e.g. gpaw_tierb_sp.out)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument("--title", default="GPAW SCF Convergence", help="Plot title")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = parse_gpaw_scf_iterations(log_path)
    if not rows:
        raise RuntimeError(f"No SCF iteration rows found in: {log_path}")

    out_path = Path(args.out) if args.out else Path("analysis/results") / f"{log_path.stem}_convergence.png"
    csv_path = Path(args.csv) if args.csv else out_path.with_suffix(".csv")

    write_csv(rows, csv_path)
    try:
        plot_rows(rows, out_path, args.title)
        plot_saved = True
    except ModuleNotFoundError as e:
        if "matplotlib" not in str(e):
            raise
        print("WARNING: matplotlib is not installed; skipping PNG plot generation.")
        print("Install with: conda activate mlip_env && conda install -c conda-forge matplotlib")
        plot_saved = False

    print(f"Parsed iterations: {len(rows)}")
    if plot_saved:
        print(f"Plot saved: {out_path}")
    else:
        print("Plot saved: (skipped)")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
