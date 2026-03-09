"""Run a single abTEM HAADF-STEM simulation from a structure file.

This is the production-style CLI equivalent of the notebook workflow in
`STEM imag simulation/abTEM_fast_scan.ipynb`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read


def _configure_abtem(device: str) -> str:
    import abtem

    if device == "cpu":
        abtem.config.set({"device": "cpu", "fft": "fftw"})
        return "cpu"

    # device=auto/gpu: try GPU; if explicitly requested, fail fast on errors.
    try:
        import cupy  # type: ignore

        _ = cupy.zeros(1)
        abtem.config.set({"device": "gpu", "fft": "cupy"})
        return "gpu"
    except Exception as e:
        if device == "gpu":
            raise RuntimeError(
                "GPU requested but CuPy/GPU backend is unavailable. "
                "Install a matching CuPy build (e.g., cupy-cuda12x) and verify CUDA access."
            ) from e
        abtem.config.set({"device": "cpu", "fft": "fftw"})
        return "cpu"


def _to_array(result) -> np.ndarray:
    arr = np.asarray(getattr(result, "array", result))
    while arr.ndim > 2:
        arr = arr[0]
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.astype(np.float64)


def _save_png(
    arr: np.ndarray,
    out_png: Path,
    title: str,
    *,
    cmap: str = "inferno",
    clip_low_pct: float = 1.0,
    clip_high_pct: float = 99.0,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Robust normalization for visibility.
    lo, hi = np.percentile(arr, clip_low_pct), np.percentile(arr, clip_high_pct)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() + 1e-12)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    fig = plt.figure(figsize=(6, 6), dpi=180)
    ax = fig.add_subplot(111)
    ax.imshow(norm, cmap=cmap, origin="lower")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="abTEM HAADF simulation (single run)")
    ap.add_argument("--structure", required=True, help="Input structure path (e.g., CIF)")
    ap.add_argument("--out-dir", default="analysis/results/stem", help="Output directory")

    ap.add_argument("--repeat", default="1,1,1", help="Repeat cell as x,y,z (e.g., 2,2,1)")
    ap.add_argument("--potential-sampling", type=float, default=0.025, help="Potential grid sampling in Angstrom")

    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--energy-kv", type=float, default=300.0, help="Accelerating voltage in kV")
    ap.add_argument("--semiangle-cutoff", type=float, default=36.0, help="Probe semiangle cutoff in mrad")
    ap.add_argument("--cs-mm", type=float, default=1.3, help="Spherical aberration Cs in mm")
    ap.add_argument("--defocus-a", type=float, default=30.0, help="Defocus in Angstrom")

    ap.add_argument("--scan-start", default="0.0,0.0", help="Fractional scan start x,y")
    ap.add_argument("--scan-end", default="1.0,1.0", help="Fractional scan end x,y")
    ap.add_argument("--scan-divisor", type=float, default=3.5, help="scan sampling = nyquist / divisor")
    ap.add_argument("--scan-sampling", type=float, default=None, help="Absolute scan sampling in Angstrom (overrides divisor)")

    ap.add_argument("--haadf-inner", type=float, default=24.0, help="HAADF detector inner angle in mrad")
    ap.add_argument("--haadf-outer", type=float, default=36.0, help="HAADF detector outer angle in mrad")
    ap.add_argument("--cmap", default="inferno", help="PNG colormap (e.g., inferno, gray, Greys_r)")
    ap.add_argument("--clip-low-pct", type=float, default=1.0, help="Lower percentile for display normalization")
    ap.add_argument("--clip-high-pct", type=float, default=99.0, help="Upper percentile for display normalization")
    ap.add_argument("--tag", default="haadf", help="Output filename tag")
    args = ap.parse_args()

    structure = Path(args.structure)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.haadf_outer > args.semiangle_cutoff:
        raise SystemExit(
            f"Invalid detector: haadf_outer={args.haadf_outer} exceeds semiangle_cutoff={args.semiangle_cutoff} mrad"
        )

    atoms = read(structure)
    rx, ry, rz = [int(x.strip()) for x in args.repeat.split(",")]
    atoms = atoms.repeat((rx, ry, rz))

    try:
        import abtem
    except ModuleNotFoundError as e:
        raise SystemExit(
            "abTEM is not installed in this environment.\n"
            "Install with:\n"
            "  pip install abtem matplotlib\n"
            "Then rerun this command.\n"
            "For a lightweight fallback image (non-abTEM), run:\n"
            "  python analysis/scripts/generate_haadf_like.py --structure dft/structures/GaN_bulk_sc_2x2x2.cif --out app/demo_data/stem/haadf_demo_latest.png"
        ) from e
    except ImportError as e:
        msg = str(e)
        if "cannot import name 'Self' from 'typing'" in msg:
            raise SystemExit(
                "abTEM import failed because this environment is Python 3.10, while the installed abTEM build expects Python >= 3.11.\n"
                "Recommended fix:\n"
                "  conda create -n stem_env python=3.11 -y\n"
                "  conda activate stem_env\n"
                "  pip install abtem matplotlib ase numpy\n"
                "Then run the STEM scripts from `stem_env`.\n"
                "Your existing `mlip_env` can remain unchanged for DFT/MLIP."
            ) from e
        raise
    from abtem import GridScan
    from abtem.detectors import AnnularDetector

    actual_device = _configure_abtem(args.device)

    potential = abtem.Potential(atoms, sampling=args.potential_sampling)
    cs_a = args.cs_mm * 1e7  # mm -> Angstrom
    probe = abtem.Probe(
        energy=args.energy_kv * 1e3,
        semiangle_cutoff=args.semiangle_cutoff,
        Cs=cs_a,
        defocus=args.defocus_a,
    )
    probe.grid.match(potential)

    start = tuple(float(x.strip()) for x in args.scan_start.split(","))
    end = tuple(float(x.strip()) for x in args.scan_end.split(","))
    sampling = args.scan_sampling
    if sampling is None:
        sampling = float(probe.aperture.nyquist_sampling) / float(args.scan_divisor)

    scan = GridScan(
        start=start,
        end=end,
        sampling=sampling,
        fractional=True,
        potential=potential,
    )

    det = AnnularDetector(inner=args.haadf_inner, outer=args.haadf_outer)
    result = probe.scan(potential, scan=scan, detectors=det).compute()
    arr = _to_array(result)

    npy_path = out_dir / f"{args.tag}.npy"
    png_path = out_dir / f"{args.tag}.png"
    json_path = out_dir / f"{args.tag}.json"

    np.save(npy_path, arr)
    _save_png(
        arr,
        png_path,
        title=(
            f"HAADF ({args.haadf_inner:.1f}-{args.haadf_outer:.1f} mrad) | "
            f"{args.energy_kv:.0f} kV | defocus {args.defocus_a:.1f} A"
        ),
        cmap=args.cmap,
        clip_low_pct=args.clip_low_pct,
        clip_high_pct=args.clip_high_pct,
    )

    try:
        fwhm = float(probe.profiles().width().compute())
    except Exception:
        fwhm = None

    meta = {
        "structure": str(structure),
        "natoms": int(len(atoms)),
        "repeat": [rx, ry, rz],
        "device": actual_device,
        "potential_sampling_A": args.potential_sampling,
        "energy_kV": args.energy_kv,
        "semiangle_cutoff_mrad": args.semiangle_cutoff,
        "Cs_mm": args.cs_mm,
        "defocus_A": args.defocus_a,
        "scan_start_frac": list(start),
        "scan_end_frac": list(end),
        "scan_sampling_A": sampling,
        "haadf_inner_mrad": args.haadf_inner,
        "haadf_outer_mrad": args.haadf_outer,
        "display_cmap": args.cmap,
        "display_clip_percentiles": [args.clip_low_pct, args.clip_high_pct],
        "probe_fwhm_A": fwhm,
        "array_shape": list(arr.shape),
        "outputs": {"png": str(png_path), "npy": str(npy_path)},
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("STEM_ABTEM_DONE")
    print("  png=", png_path)
    print("  npy=", npy_path)
    print("  meta=", json_path)
    print("  device=", actual_device)
    print("  shape=", arr.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
