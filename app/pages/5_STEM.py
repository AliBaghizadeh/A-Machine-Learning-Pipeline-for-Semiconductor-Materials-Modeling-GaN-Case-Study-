from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.lib.demo_io import pretty_path


st.set_page_config(page_title="🔬 STEM Image Demo", layout="wide")

root = Path(__file__).resolve().parents[2]
stem_dir = root / "app" / "demo_data" / "stem"
stem_dir.mkdir(parents=True, exist_ok=True)


def _find_images() -> list[Path]:
    images: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(sorted(stem_dir.glob(ext)))
    return images


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _cif_candidates() -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(sorted((root / "analysis" / "results" / "large_scale_mlip").glob("*_relaxed.cif")))
    candidates.extend(sorted((root / "analysis" / "results" / "large_scale_mlip_demo").glob("*_relaxed.cif")))
    candidates.extend(sorted((root / "dft" / "structures").glob("*.cif")))
    # De-duplicate while preserving order.
    seen = set()
    out: list[Path] = []
    for p in candidates:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        out.append(p)
    return out


def _scenario_catalog() -> dict[str, dict]:
    return {
        "Bulk GaN (relaxed)": {
            "description": "Reference crystalline GaN example. Useful baseline for comparing contrast patterns.",
            "cif_paths": [
                root / "analysis" / "results" / "large_scale_mlip" / "GaN_bulk_sc_6x6x4_relaxed__mlip_relaxed.cif",
                root / "dft" / "structures" / "GaN_bulk_sc_4x4x4_relaxed.cif",
                root / "dft" / "structures" / "GaN_bulk_sc_2x2x2.cif",
            ],
            # Put your confirmed showcase output first.
            "image_keywords": ["haadf_xxl_fast_333", "bulk", "gan_bulk"],
        },
        "Vacancy defect": {
            "description": "Point-defect example where one atom is missing. Shows local structural change around a defect.",
            "cif_paths": [
                root / "analysis" / "results" / "large_scale_mlip" / "GaN_V_Ga_sc_4x4x4_relaxed__mlip_relaxed.cif",
                root / "dft" / "structures" / "GaN_V_Ga_sc_4x4x4_relaxed.cif",
                root / "dft" / "structures" / "GaN_V_N_sc_4x4x4_relaxed.cif",
            ],
            "image_keywords": ["vacancy", "v_ga", "v_n"],
        },
        "Vacancy-line defect": {
            "description": "Extended defect-line example. Useful for checking whether larger defect features remain visible in simulated imaging.",
            "cif_paths": [
                root / "analysis" / "results" / "final_structures" / "GaN_vacancy_line_N_sc_4x4x4_relaxed__DFT_SR_final.cif",
                root / "dft" / "structures" / "GaN_vacancy_line_N_sc_4x4x4_relaxed.cif",
                root / "dft" / "structures" / "GaN_defect_line_V_N_sc_3x3x2__strain00.cif",
            ],
            "image_keywords": ["line", "vacancy_line", "defect_line"],
        },
        "Custom: select a CIF from repo": {
            "description": "Choose any available CIF in this repo and inspect matching precomputed images (if available).",
            "cif_paths": [],
            "image_keywords": [],
        },
    }


def _match_images(images: list[Path], keywords: list[str], *, fallback_n: int = 4) -> list[Path]:
    if not images:
        return []
    kws = [k.lower() for k in keywords if k]
    if kws:
        hits = [p for p in images if any(k in p.stem.lower() for k in kws)]
        if hits:
            return hits
    return images[:fallback_n]


st.title("From atomic structure -> microscopy-like image")
st.markdown(
    "This page is a **demo-only visualization** that links atomic structures to microscopy-like outputs."
)
st.markdown(
    "1. We generate/relax structures using MLIP (fast).  \n"
    "2. We can convert structures into simulated STEM images (abTEM).  \n"
    "3. These images are **precomputed** for this demo."
)
st.info(
    "Why this page exists: it shows that MLIP-relaxed structures can be used downstream to generate microscopy-like observables.\n\n"
    "Privacy: all assets are local; the demo makes no external calls."
)

st.markdown(
    "**Plain-language definition:** Scanning Transmission Electron Microscopy (STEM) is an imaging technique that can show atomic columns. "
    "A simulated STEM image helps connect simulated structures to something similar to an experimental measurement."
)

st.divider()

st.subheader("Choose an example")
catalog = _scenario_catalog()
scenario = st.selectbox("Example scenario", list(catalog.keys()))
entry = catalog[scenario]

all_candidates = _cif_candidates()
selected_cif: Path | None
if scenario == "Custom: select a CIF from repo":
    if all_candidates:
        selected_cif = st.selectbox(
            "Select CIF file",
            all_candidates,
            format_func=lambda p: pretty_path(p, root=root),
        )
    else:
        selected_cif = None
else:
    selected_cif = _first_existing(entry["cif_paths"])

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("**What**")
    st.write(entry["description"])
    st.markdown("**Structure path**")
    if selected_cif is not None:
        st.code(pretty_path(selected_cif, root=root), language="text")
    else:
        st.info("No matching CIF found for this scenario.")

    st.markdown("**Why**")
    st.write(
        "Use this to judge whether defect signatures could be visible and to create synthetic image data for ML workflows."
    )

with right:
    st.markdown("**Output (precomputed images)**")
    images = _find_images()
    matched = _match_images(images, entry.get("image_keywords") or [])
    if not images:
        st.info("No precomputed STEM images found. To add them, copy PNG/JPG files into `app/demo_data/stem/`.")
    else:
        if not matched:
            st.info("No scenario-specific image tags matched. Showing a small fallback sample.")
            matched = images[:4]
        for p in matched[:6]:
            st.image(str(p), caption=pretty_path(p, root=root), width="stretch")
        if len(matched) > 6:
            with st.expander("More images"):
                for p in matched[6:]:
                    st.image(str(p), caption=pretty_path(p, root=root), width="stretch")

st.divider()

with st.expander("How STEM images are produced (optional, not executed in demo)", expanded=False):
    cmd = (
        "python analysis/scripts/stem_abtem_haadf.py "
        "--structure dft/structures/GaN_bulk_sc_2x2x2.cif "
        "--out-dir app/demo_data/stem "
        "--repeat 3,3,3 --scan-start 0.0,0.0 --scan-end 0.35,0.35 --scan-divisor 16 "
        "--potential-sampling 0.10 --energy-kv 300 --semiangle-cutoff 80 --haadf-inner 50 --haadf-outer 75 "
        "--defocus-a 0 --device gpu --cmap gray --tag haadf_xxl_fast_333"
    )
    st.caption("This app page does not run the command below. It is shown for reproducibility only.")
    st.code(cmd, language="bash")
    st.write("If needed, run abTEM in a separate environment (for example `stem_env`).")

with st.expander("What do the advanced STEM parameters mean?", expanded=False):
    st.markdown(
        "- **Defocus:** how focus is shifted; changes image contrast.\n"
        "- **Detector angles (inner/outer):** which scattered electrons are counted.\n"
        "- **Probe energy (kV):** electron beam energy.\n"
        "- **Sampling:** simulation pixel spacing vs speed.\n"
        "- **Repeat cell:** tiles the structure for a larger field of view."
    )
