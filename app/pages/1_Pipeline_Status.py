from __future__ import annotations

import streamlit as st

from app.lib.demo_io import get_demo_dir, load_manifest, path_exists, pretty_path

# UI updates in this revision:
# 1) Reorganized page into: what this demo is, what this run uses, and health + quick stats.
# 2) Added a one-minute summary box and a non-expert glossary near the top.
# 3) Replaced dense overview text with a simple pipeline flow and plain-language labels.
# 4) Converted artifact display to 3 compact cards (model/dataset/reference) with status badges.
# 5) Replaced health dataframe with compact status metrics and actionable error guidance.
# 6) Simplified dataset-source labels in main UI; moved original tier labels to advanced section.
# 7) Kept all existing data outputs (manifest stats, artifact table, health logic, bundle count, raw metadata).


st.set_page_config(page_title="Pipeline Status", layout="wide")
st.title("Pipeline Status")
st.caption("Checks that the demo bundle contains a model + dataset + reference results.")
st.info("This page does not run DFT. It only verifies the demo bundle and shows what the app will use.")

demo_dir = get_demo_dir()
m = load_manifest(demo_dir)

frozen = m.get("frozen") or {}
stats = (m.get("dataset") or {}).get("dataset_stats") or {}
copied = m.get("copied_files") or []

st.subheader("A) What This Demo Is")
st.success(
    "One-minute read: this tab confirms the app has everything needed to run the offline demo "
    "(model, training dataset, and reference results). If these checks pass, the rest of the app "
    "can show relaxation and validation outputs reliably."
)
with st.expander("Glossary (non-expert)"):
    st.markdown(
        "- **DFT (Density Functional Theory):** high-accuracy reference simulation used to create training labels.\n"
        "- **ML potential:** a fast machine-learned model that predicts energies and forces.\n"
        "- **Energy/force evaluation:** computing energy and forces for a fixed structure.\n"
        "- **Short geometry relaxation:** a small number of optimization steps to improve atom positions.\n"
        "- **Structure:** one atomic configuration used as a training example."
    )

st.markdown(
    "**Crystal structure -> Reference simulation (DFT) -> Training dataset -> ML potential -> "
    "Large-structure relaxation -> Validation checks**"
)
st.caption("In this demo, the heavy simulation is precomputed and packaged.")

st.divider()

st.subheader("B) What This Run Uses")
st.markdown("**Artifacts Used in This Demo Run**")
st.caption(
    "These files define the exact model and dataset used for the demonstration. "
    "Keeping these references fixed ensures the demo is reproducible."
)


def _ok_label(ok: bool) -> str:
    return "✅ Available" if ok else "❌ Missing"


model_ok = path_exists(frozen.get("model_path"))
dataset_ok = path_exists(frozen.get("dataset_dir"))
dft_ok = path_exists(frozen.get("dft_json"))

col_m, col_d, col_r = st.columns(3)
with col_m:
    st.markdown(f"**ML potential model**  \n{_ok_label(model_ok)}")
    st.caption("Model used by the demo for prediction and relaxation.")
    st.code(pretty_path(frozen.get("model_path")) or "missing", language="text")
with col_d:
    st.markdown(f"**Training dataset**  \n{_ok_label(dataset_ok)}")
    st.caption("DFT-labeled structures used to train the ML potential.")
    st.code(pretty_path(frozen.get("dataset_dir")) or "missing", language="text")
with col_r:
    st.markdown(f"**Reference results (DFT)**  \n{_ok_label(dft_ok)}")
    st.caption("Reference energy/force results used for validation checks.")
    st.code(pretty_path(frozen.get("dft_json")) or "missing", language="text")

with st.expander("Show as table"):
    artifact_rows = [
        {"Artifact": "ML Potential Model", "Location": pretty_path(frozen.get("model_path"))},
        {"Artifact": "Training Dataset", "Location": pretty_path(frozen.get("dataset_dir"))},
        {"Artifact": "Reference DFT Results", "Location": pretty_path(frozen.get("dft_json"))},
    ]
    st.dataframe(artifact_rows, width="stretch", hide_index=True)

st.divider()

st.subheader("C) Health + Quick Stats")
st.markdown("**Pipeline Health Check**")
st.caption("The demo verifies that required files exist before visualization and analysis steps run.")
h1, h2, h3 = st.columns(3)
with h1:
    st.metric("Model", "Available" if model_ok else "Missing")
with h2:
    st.metric("Dataset", "Available" if dataset_ok else "Missing")
with h3:
    st.metric("Reference results", "Available" if dft_ok else "Missing")

if not (model_ok and dataset_ok and dft_ok):
    st.error("Some required artifacts are missing. Next action: run `make demo-artifacts`.")

st.divider()

st.markdown("**Training Dataset Summary**")
st.caption(
    "A structure is one atomic configuration used for training. "
    "'Filtered' means size-limited for training speed; 'full dataset' includes all extracted structures."
)
summary_rows = [
    {"Metric": "Total structures (filtered)", "Value": stats.get("total_structures", "unknown")},
    {"Metric": "Total structures (full dataset)", "Value": stats.get("total_structures_full", "unknown")},
    {"Metric": "Train set", "Value": stats.get("train", "unknown")},
    {"Metric": "Validation set", "Value": stats.get("val", "unknown")},
    {"Metric": "Test set", "Value": stats.get("test", "unknown")},
    {"Metric": "Maximum atoms allowed", "Value": stats.get("max_atoms_filter", "unknown")},
]
st.dataframe(summary_rows, width="stretch", hide_index=True)

st.markdown("**Structure Sources**")
sources = stats.get("sources") or {}
source_rows = [
    {"Source": "DFT energy+forces (single-step)", "Count": sources.get("tier_b_sp", 0)},
    {"Source": "DFT short relaxation (few steps)", "Count": sources.get("tier_b_sr", 0)},
    {"Source": "Legacy / other source", "Count": sources.get("tier_a", 0)},
]
st.dataframe(source_rows, width="stretch", hide_index=True)

st.divider()

st.subheader("Demo Artifact Bundle")
st.info(
    "To keep the demo lightweight and reproducible, a small subset of project outputs is packaged "
    "into a golden-run bundle and loaded from the demo directory."
)
st.markdown(
    "- trained ML potential\n"
    "- dataset snapshot\n"
    "- validation metrics\n"
    "- example structures\n"
    "- configuration metadata"
)
st.write("Bundle path used by app:", pretty_path(demo_dir))
st.write("Bundled files:", len(copied))

st.divider()

with st.expander("Technical Details (Advanced)"):
    st.caption("This is raw metadata for debugging.")
    st.markdown("**Dataset metadata JSON**")
    st.json(stats)
    st.markdown(
        "**Original source labels (raw):**\n"
        "- `tier_b_sp`: DFT energy+forces (single-step)\n"
        "- `tier_b_sr`: DFT short relaxation (few steps)\n"
        "- `tier_a`: Legacy / other source"
    )
    if copied:
        cleaned = []
        for row in copied:
            if not isinstance(row, dict):
                continue
            cleaned.append(
                {
                    "dst": pretty_path(row.get("dst")),
                    "src": pretty_path(row.get("src")),
                    "size_bytes": row.get("size_bytes"),
                }
            )
        st.markdown("**Bundled file list**")
        st.dataframe(cleaned, width="stretch")
