"""Generate a lightweight HAADF-like projection image from a structure.

This is a demo fallback for environments without abTEM.
It projects atomic columns and weights intensity by Z^1.7, then applies a
small Gaussian-like blur.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _blur2d(img: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    if sigma <= 0:
        return img
    radius = max(1, int(3.0 * sigma))
    k = _gaussian_kernel1d(sigma=sigma, radius=radius)

    # Separable convolution: x then y.
    pad = radius
    xpad = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    out_x = np.zeros_like(img)
    for i in range(img.shape[1]):
        out_x[:, i] = (xpad[:, i : i + 2 * pad + 1] * k[None, :]).sum(axis=1)

    ypad = np.pad(out_x, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for j in range(img.shape[0]):
        out[j, :] = (ypad[j : j + 2 * pad + 1, :] * k[:, None]).sum(axis=0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, help="Path to CIF/XYZ/TRAJ structure")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--bins", type=int, default=384, help="Image resolution")
    ap.add_argument("--sigma", type=float, default=1.4, help="Blur sigma in pixels")
    args = ap.parse_args()

    a = read(args.structure)
    pos = a.get_positions()
    z = a.get_atomic_numbers().astype(float)

    # Project along z-axis to x-y detector plane.
    x = pos[:, 0]
    y = pos[:, 1]
    w = np.power(z, 1.7)

    # Histogram bounds with a small margin.
    mx = 0.05 * (x.max() - x.min() + 1e-9)
    my = 0.05 * (y.max() - y.min() + 1e-9)
    xr = (x.min() - mx, x.max() + mx)
    yr = (y.min() - my, y.max() + my)

    h, _, _ = np.histogram2d(y, x, bins=args.bins, range=[yr, xr], weights=w)
    h = _blur2d(h, sigma=args.sigma)

    # Normalize to [0, 1].
    h = h - h.min()
    if h.max() > 0:
        h = h / h.max()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg", force=True)

    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(111)
    ax.imshow(h, cmap="inferno", origin="lower")
    ax.set_axis_off()
    ax.set_title("HAADF-like projection (demo fallback)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"HAADF_DEMO_DONE out={out} atoms={len(a)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
