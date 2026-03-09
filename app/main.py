from __future__ import annotations

import base64
from pathlib import Path


def _render_svg(path: Path) -> str:
    b = path.read_bytes()
    enc = base64.b64encode(b).decode("ascii")
    return f"<img alt='{path.name}' style='max-width: 100%; height: auto;' src='data:image/svg+xml;base64,{enc}'/>"


def main():
    # Import Streamlit only when executing the app, so importing this module
    # in tests does not require streamlit to be installed.
    import streamlit as st

    from app.lib.demo_io import get_demo_dir, load_manifest, read_text_if_exists

    st.set_page_config(page_title="GaN DFT + MLIP Pipeline Demo", layout="wide")
    root = Path(__file__).resolve().parents[1]
    demo_dir = get_demo_dir()

    st.title("GaN DFT + MLIP Pipeline Demo")
    st.caption("Offline Materials AI Pipeline Demonstration")
    st.markdown(
        "This demo showcases a minimal closed-loop workflow combining DFT simulations, "
        "machine-learned interatomic potentials (MLIPs), and simple validation gates.\n\n"
        "The goal is to demonstrate how a small set of high-accuracy DFT calculations can train a fast ML model "
        "capable of exploring larger atomic systems."
    )

    st.subheader("🧠 Pipeline Overview")
    # Prefer the user-provided PNG (demo_data), fall back to the built-in SVG.
    pipeline_png = root / "app" / "demo_data" / "plots" / "Pipleline.png"
    if pipeline_png.exists():
        st.image(str(pipeline_png), width="stretch")
    else:
        svg = Path(__file__).resolve().parent / "assets" / "pipeline_diagram.svg"
        if svg.exists():
            st.components.v1.html(_render_svg(svg), height=380, scrolling=False)
        else:
            st.info("Missing pipeline diagram.")
    st.caption("This demo UI reads precomputed artifacts and does not run DFT.")

    st.subheader("Key Concepts")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**DFT (Density Functional Theory)**\n\nQuantum-mechanical simulations used to compute accurate energies and forces.")
    with c2:
        st.markdown(
            "**MLIP (Machine-Learned Interatomic Potential)**\n\n"
            "A neural potential trained on DFT data that predicts energies and forces much faster."
        )
    with c3:
        st.markdown(
            "**Validation Gates**\n\n"
            "Spot-checks comparing MLIP predictions to DFT results to ensure the model is usable for the next step."
        )

    m = load_manifest(demo_dir)

    st.subheader("⚡ Demo Mode Summary")
    gates = m.get("gates") or {}
    eg = gates.get("energy_gate") or {}
    fg = gates.get("force_gate") or {}

    summary_rows = [
        {"Item": "Demo Mode", "Status": str(bool(m.get("demo", True)))},
        {"Item": "Created", "Status": str(m.get("created") or "unknown")},
        {"Item": "Energy Gate", "Status": "Passed" if eg.get("pass") else "Failed"},
        {"Item": "Force Gate", "Status": "Passed" if fg.get("pass") else "Failed"},
        {"Item": "Energy Threshold", "Status": f"{eg.get('threshold_eV_per_atom')} eV / atom" if eg.get("threshold_eV_per_atom") is not None else "unknown"},
    ]
    st.dataframe(summary_rows, width="stretch", hide_index=True)

    st.divider()

    st.subheader("🔒 Privacy and Offline Operation")
    st.markdown("This demo runs fully locally by default.")
    st.markdown("**What the demo does**")
    st.markdown(
        "- Reads locally generated DFT and MLIP artifacts\n"
        "- Loads precomputed demo artifacts from `app/demo_data/`\n"
        "- Runs a local Streamlit interface\n"
        "- Optionally runs a local RAG assistant (Ollama) or a cloud assistant (OpenAI) if you enable it"
    )
    st.markdown("**What the demo does NOT do (default)**")
    st.markdown(
        "- No external API calls\n"
        "- No telemetry\n"
        "- No cloud uploads\n"
        "- No remote vector databases\n"
        "- No DFT execution from the UI"
    )
    privacy = read_text_if_exists(Path(__file__).resolve().parents[1] / "docs" / "privacy.md")
    if privacy:
        with st.expander("Privacy details"):
            st.markdown(privacy)

    st.subheader("📍 Data Locations (Repo-Relative)")
    loc_rows = [
        {"Data": "DFT Results", "Location": "dft/results/"},
        {"Data": "MLIP Datasets", "Location": "mlip/data/datasets/"},
        {"Data": "Training Runs", "Location": "mlip/results/"},
        {"Data": "Demo Artifacts", "Location": "analysis/artifacts/"},
        {"Data": "RAG Papers", "Location": "literature/ (or rag/sources/)"},
    ]
    st.dataframe(loc_rows, width="stretch", hide_index=True)

    arch = read_text_if_exists(Path(__file__).resolve().parents[1] / "docs" / "architecture.md")
    if arch:
        with st.expander("📚 Architecture details"):
            st.markdown(arch)

    st.info("Tip: run `make demo-artifacts` to refresh app/demo_data from your latest local runs.")


if __name__ == "__main__":
    main()
