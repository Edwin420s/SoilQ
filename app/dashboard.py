"""
SoilQ - Enterprise Quantum-Assisted Agronomy Dashboard
Qiskit Fall Fest Nairobi 2026 - Challenge 08: Quantum AI Classifier

Tabs
----
1. Field Predictor and Agronomic Advisory
2. Exploratory Data Analysis (EDA)
3. Quantum Circuit Studio
4. Benchmark and Model Performance
5. Dataset Ingestion and Audit
6. Methodology and Ethics
"""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from soilq.config import PROJECT_ROOT
from soilq.data import audit_dataset
from soilq.pipeline import predict_from_model_dir, train_benchmark
from soilq.prepare import (
    DEFAULT_FEATURES,
    DEFAULT_TARGET,
    SOIL_FEATURES,
    load_prepared,
    prepare,
)

# Page Configuration
st.set_page_config(
    page_title="SoilQ - Quantum Agronomy AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("SoilQ - Quantum-Assisted Maize Agronomy")
    st.caption(
        "Machine Learning and Qiskit Variational Classifier Platform - "
        "Kenyan Field Trial Experiment (Chuka and Kandara) - Qiskit Fall Fest Nairobi 2026"
    )
with col_h2:
    st.markdown(
        """
        <div style="text-align: right; margin-top: 10px;">
            <span style="background: #10b981; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">Qiskit 2.6.1</span>
            <span style="background: #3b82f6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">Python 3.14</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Navigation Tabs
tabs = st.tabs([
    "Field Predictor and Advisory",
    "Exploratory Data Analysis",
    "Quantum Circuit Studio",
    "Model Benchmarking",
    "Dataset Audit and Ingest",
    "Methodology and Research",
])

# -----------------------------------------------------------------------------
# TAB 1: Field Predictor & Agronomic Advisory
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Interactive Field Predictor and Agronomic Advisor")
    st.markdown(
        "Input agronomic measurements from an on-station or smallholder maize plot "
        "to generate **simultaneous predictions** across Classical ML, Variational QML (VQC), and Quantum Kernel (QSVM) models."
    )

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        st.subheader("Vegetative Growth")
        h6 = st.slider("Plant Height at 6th-Leaf (cm)", 10.0, 65.0, 32.5, 0.5)
        h10 = st.slider("Plant Height at 10th-Leaf (cm)", 30.0, 300.0, 145.0, 1.0)
    with col_in2:
        st.subheader("Chlorophyll / Nitrogen Status")
        spad6 = st.slider("SPAD Chlorophyll (6th-Leaf)", 15.0, 65.0, 38.5, 0.5)
        spad10 = st.slider("SPAD Chlorophyll (10th-Leaf)", 20.0, 60.0, 42.0, 0.5)
    with col_in3:
        st.subheader("Biomass and Management")
        stover = st.slider("Stover Biomass Yield (Mg/ha)", 0.5, 15.0, 4.5, 0.1)
        model_dir_input = st.text_input("Trained Model Directory", "models/experiment_001")

    input_sample = {
        "Height_6th": h6,
        "Height_10th": h10,
        "SPAD_6th": spad6,
        "SPAD_10th": spad10,
        "Stover_Mg/ha": stover,
    }

    if st.button("Run Multi-Model Quantum and Classical Inference", key="btn_predict", use_container_width=True):
        mdir_path = PROJECT_ROOT / model_dir_input
        if not mdir_path.exists():
            st.error(f"Model directory '{model_dir_input}' does not exist. Train models first in the 'Model Benchmarking' tab.")
        else:
            with st.spinner("Executing quantum statevector simulation and classical inference ..."):
                try:
                    preds = predict_from_model_dir(mdir_path, input_sample)

                    st.markdown("### Multi-Model Prediction Results")
                    cols = st.columns(len(preds))

                    positive_votes = 0
                    model_names_map = {
                        "logisticregression": "Logistic Regression",
                        "rbf_svm": "RBF Support Vector",
                        "randomforest": "Random Forest",
                        "qiskit_variational_classifier": "Qiskit Variational (VQC)",
                        "quantum_kernel_classifier": "Quantum Kernel (QSVM)",
                    }

                    for i, (m_id, res) in enumerate(preds.items()):
                        m_label = model_names_map.get(m_id, m_id)
                        p_val = res["prediction"]
                        prob = res["probability_positive"]
                        conf = res["confidence"] * 100
                        if p_val == 1:
                            positive_votes += 1

                        with cols[i]:
                            st.markdown(f"**{m_label}**")
                            if p_val == 1:
                                st.success("HIGH YIELD (>0.74 Mg/ha)")
                            else:
                                st.warning("LOW YIELD (<=0.74 Mg/ha)")
                            st.metric("P(High Yield)", f"{prob * 100:.1f}%")
                            st.caption(f"Confidence: {conf:.1f}%")

                    # Ensemble consensus
                    consensus_pct = (positive_votes / len(preds)) * 100
                    st.divider()
                    st.subheader("Ensemble Consensus")
                    st.progress(
                        positive_votes / len(preds),
                        text=f"{positive_votes}/{len(preds)} models agree on High Yield ({consensus_pct:.0f}% agreement)",
                    )

                    # Radar comparison
                    st.subheader("Agronomic Feature Profile vs Dataset Median")
                    categories = ["Height 6th", "Height 10th", "SPAD 6th", "SPAD 10th", "Stover Yield"]
                    ref_medians = [27.5, 80.5, 38.2, 39.7, 4.29]
                    user_vals = [h6, h10, spad6, spad10, stover]
                    norm_user = [u / m for u, m in zip(user_vals, ref_medians)]
                    norm_ref = [1.0] * 5

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(r=norm_user + [norm_user[0]], theta=categories + [categories[0]], fill='toself', name='Current Plot Input'))
                    fig_radar.add_trace(go.Scatterpolar(r=norm_ref + [norm_ref[0]], theta=categories + [categories[0]], line=dict(dash='dash', color='gray'), name='Dataset Median Baseline'))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2.5])), showlegend=True, height=380)
                    st.plotly_chart(fig_radar, use_container_width=True)

                    # Advisory insights
                    st.subheader("Agronomic Advisory Insights")
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        st.info(
                            f"**Nitrogen / Chlorophyll Status**: SPAD reading at 10th leaf is **{spad10:.1f}**. "
                            f"{'Sufficient leaf greenness indicating adequate N uptake.' if spad10 >= 38.0 else 'Sub-optimal SPAD indicates potential nitrogen deficiency; consider targeted top-dressing.'}"
                        )
                    with adv_col2:
                        st.info(
                            f"**Biomass Conversion**: Stover biomass of **{stover:.1f} Mg/ha** with height **{h10:.1f} cm** "
                            f"{'reflects strong vegetative development supporting above-median grain filling.' if stover >= 4.0 else 'suggests stunted vegetative growth, likely constrained by early water deficit.'}"
                        )

                except Exception as exc:
                    st.error(f"Inference error: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

# -----------------------------------------------------------------------------
# TAB 2: Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown(
        "Interactive analysis of the Kenyan maize field experiment across "
        "**Chuka** and **Kandara** sites over 4 seasons (`LR16`, `LR17`, `SR16`, `SR17`)."
    )

    try:
        merged_df = load_prepared()
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Plots Analyzed", len(merged_df))
        col_m2.metric("Median Maize Yield", f"{merged_df['Yield_Mg/ha'].median():.3f} Mg/ha")
        col_m3.metric("Sites", f"{merged_df['Site'].nunique()} (Chuka, Kandara)")
        col_m4.metric("Seasons", f"{merged_df['Season'].nunique()} (LR16-SR17)")

        st.divider()
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Yield and Tillage Distributions", "Soil Input Efficacy", "Feature Correlation Heatmap"])

        with eda_tab1:
            col_eda1, col_eda2 = st.columns(2)
            with col_eda1:
                fig_box = px.box(
                    merged_df,
                    x="Site",
                    y="Yield_Mg/ha",
                    color="Tillage",
                    points="all",
                    title="Maize Grain Yield by Site and Tillage Depth (D0: No-Till vs D15: Conventional 15cm)",
                    labels={"Yield_Mg/ha": "Grain Yield (Mg/ha)"},
                    color_discrete_sequence=["#10b981", "#3b82f6"],
                )
                st.plotly_chart(fig_box, use_container_width=True)
            with col_eda2:
                fig_hist = px.histogram(
                    merged_df,
                    x="Yield_Mg/ha",
                    color="high_yield",
                    nbins=30,
                    marginal="rug",
                    title="Empirical Yield Distribution and Median Threshold (0.735 Mg/ha)",
                    labels={"high_yield": "Class (1=High, 0=Low)"},
                    color_discrete_sequence=["#f59e0b", "#10b981"],
                )
                fig_hist.add_vline(x=0.735, line_dash="dash", line_color="red", annotation_text="Median Split")
                st.plotly_chart(fig_hist, use_container_width=True)

        with eda_tab2:
            fig_soil = px.bar(
                merged_df.groupby(["Soil_Input", "Site"], as_index=False)["Yield_Mg/ha"].mean(),
                x="Soil_Input",
                y="Yield_Mg/ha",
                color="Site",
                barmode="group",
                title="Mean Maize Yield by Soil Input Treatment Across Sites",
                labels={"Soil_Input": "Soil Input (C=Control, F=Fertilizer, M=Manure, R=Residue)", "Yield_Mg/ha": "Mean Yield (Mg/ha)"},
            )
            st.plotly_chart(fig_soil, use_container_width=True)
            st.caption("Treatments: C (Control), F (Inorganic Fertilizer), RF (Residue+Fertilizer), RFM (Residue+Fertilizer+Manure), RML (Residue+Manure+Legume), RTiM (Tithonia+Manure), RTiP (Tithonia+Phosphate).")

        with eda_tab3:
            num_cols = ["Yield_Mg/ha", "Stover_Mg/ha", "Height_6th", "Height_10th", "SPAD_6th", "SPAD_10th", "SWC_10cm", "SWC_20cm"]
            corr = merged_df[num_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Agronomic Pearson Correlation Matrix",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not load EDA dataset: {exc}")

# -----------------------------------------------------------------------------
# TAB 3: Quantum Circuit Studio
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Quantum Circuit Studio and State Simulator")
    st.markdown(
        "Interactive Qiskit circuit architect. Customize qubits, ansatz depth, "
        "entanglement topologies, and visualize the resulting quantum statevector."
    )

    qc_col1, qc_col2, qc_col3, qc_col4 = st.columns(4)
    with qc_col1:
        n_q = st.slider("Qubits (Features)", 1, 6, 4)
    with qc_col2:
        reps = st.slider("Ansatz Layers (reps)", 1, 4, 2)
    with qc_col3:
        ent = st.selectbox("Entanglement Topology", ["ring", "linear", "full"])
    with qc_col4:
        fmap = st.selectbox("Feature Map", ["angle", "zz"])

    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from soilq.quantum import build_feature_map, build_variational_ansatz

        rng = np.random.default_rng(42)
        demo_x = rng.uniform(-0.8, 0.8, n_q)
        demo_theta = rng.uniform(-0.5, 0.5, reps * n_q * 2)

        qc_fmap = build_feature_map(demo_x, n_q, fmap)
        qc_ansatz = build_variational_ansatz(n_q, demo_theta, reps, ent)

        qc_full = QuantumCircuit(n_q)
        qc_full.compose(qc_fmap, inplace=True)
        qc_full.barrier(label="Encoding")
        qc_full.compose(qc_ansatz, inplace=True)
        qc_full.barrier(label="Variational")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Qubits", n_q)
        col_m2.metric("Circuit Depth", qc_full.depth())
        col_m3.metric("Total Gates", sum(qc_full.count_ops().values()))
        col_m4.metric("Trainable Parameters", len(demo_theta))

        st.subheader("Quantum Circuit Diagram (ASCII)")
        st.code(qc_full.draw(output="text", fold=80).__str__(), language="text")

        st.subheader("Quantum Statevector Probabilities (Sample Input)")
        sv = Statevector.from_instruction(qc_full)
        probs = sv.probabilities()
        labels = [format(i, f"0{n_q}b") for i in range(2 ** n_q)]

        prob_df = pd.DataFrame({"Basis State": labels, "Probability": probs})
        fig_sv = px.bar(
            prob_df,
            x="Basis State",
            y="Probability",
            title=f"Measurement Probability Distribution Across {2 ** n_q} Quantum States",
            color="Probability",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_sv, use_container_width=True)

        exp_z0 = float(probs[0::2].sum() - probs[1::2].sum())
        p_high = 1.0 / (1.0 + np.exp(-4.0 * exp_z0))
        st.info(f"Pauli-Z Expectation on Qubit 0: **{exp_z0:.4f}** Transformed Sigmoid Probability P(High Yield): **{p_high * 100:.1f}%**")

    except Exception as exc:
        st.error(f"Circuit error: {exc}")

# -----------------------------------------------------------------------------
# TAB 4: Model Benchmarking
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Model Benchmarking and Performance Comparison")
    st.markdown(
        "Compare Classical Machine Learning against Variational Quantum Classifiers (VQC) "
        "and Quantum Support Vector Machines (QSVM) on identical held-out test sets."
    )

    merged_csv_default = str(PROJECT_ROOT / "data" / "processed" / "kenyan_maize_merged.csv")
    input_file_train = st.text_input("Dataset CSV File", merged_csv_default)

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        target_train = st.text_input("Target Column", DEFAULT_TARGET)
        test_split = st.slider("Holdout Test Split", 0.1, 0.4, 0.25, 0.05)
    with b_col2:
        q_iters = st.slider("VQC Max Iterations (COBYLA)", 10, 100, 40, 5)
        run_qkernel = st.checkbox("Include Quantum Kernel (QSVM)", value=True)
    with b_col3:
        out_exp_dir = st.text_input("Output Directory", "models/experiment_001")
        rep_exp_path = st.text_input("Report Path", "data/reports/experiment_001.json")

    if st.button("Run Full Benchmark Experiment", key="btn_run_bench", use_container_width=True):
        if not Path(input_file_train).exists():
            st.error(f"Input file '{input_file_train}' not found. Run data prep first.")
        else:
            prog = st.progress(0, text="Training models ...")
            try:
                prog.progress(10, text="Preprocessing dataset ...")
                rep = train_benchmark(
                    input_path=input_file_train,
                    target=target_train,
                    features=DEFAULT_FEATURES,
                    output_dir=out_exp_dir,
                    report_path=rep_exp_path,
                    test_size=test_split,
                    quantum_maxiter=q_iters,
                    include_quantum_kernel=run_qkernel,
                )
                prog.progress(100, text="Benchmark complete")
                st.success(f"Benchmarking completed! Saved report to `{rep['report_path']}`")

            except Exception as exc:
                prog.empty()
                st.error(f"Training failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    rep_file_check = PROJECT_ROOT / rep_exp_path
    if rep_file_check.exists():
        st.divider()
        st.subheader("Latest Benchmark Results")
        try:
            report_data = json.loads(rep_file_check.read_text(encoding="utf-8"))
            models_dict = report_data.get("models", {})

            rows = []
            for m_name, m_data in models_dict.items():
                rows.append({
                    "Model": m_name,
                    "Accuracy": round(m_data.get("accuracy", 0), 4),
                    "Balanced Acc": round(m_data.get("balanced_accuracy", 0), 4),
                    "F1 Score": round(m_data.get("f1", 0), 4),
                    "ROC-AUC": round(m_data.get("roc_auc") or 0.0, 4),
                    "Brier Score": round(m_data.get("brier_score") or 0.0, 4),
                    "Runtime (s)": round(m_data.get("fit_seconds", 0), 2),
                })
            df_results = pd.DataFrame(rows)
            st.dataframe(df_results, use_container_width=True)

            fig_bar = px.bar(
                df_results,
                x="Model",
                y=["F1 Score", "ROC-AUC", "Accuracy"],
                barmode="group",
                title="Model Performance Metric Comparison",
                color_discrete_sequence=["#10b981", "#3b82f6", "#8b5cf6"],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Confusion Matrix Heatmaps")
            cm_cols = st.columns(len(models_dict))
            for i, (m_name, m_data) in enumerate(models_dict.items()):
                cm = m_data.get("confusion_matrix", [[0, 0], [0, 0]])
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=["Pred Low", "Pred High"],
                    y=["True Low", "True High"],
                    color_continuous_scale="Blues",
                    title=m_name,
                )
                with cm_cols[i]:
                    st.plotly_chart(fig_cm, use_container_width=True)

        except Exception as exc:
            st.error(f"Could not parse report: {exc}")

# -----------------------------------------------------------------------------
# TAB 5: Dataset Ingestion & Audit
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Dataset Ingestion, Audit and Integrity Verification")
    st.markdown(
        "Audit raw experimental agronomy CSVs, verify SHA-256 provenance hashes, "
        "and run plot-level data consolidation."
    )

    col_aud1, col_aud2 = st.columns([1, 1])
    with col_aud1:
        if st.button("Re-run Data Merge and Pipeline Ingestion", key="btn_ingest"):
            with st.spinner("Consolidating tables ..."):
                try:
                    merged, prov = prepare(save=True)
                    st.success(f"Merged {len(merged)} rows x {len(merged.columns)} columns.")
                    st.json(prov["class_counts"])
                except Exception as exc:
                    st.error(str(exc))

    with col_aud2:
        raw_dir = st.text_input("Audit Directory", str(PROJECT_ROOT / "data" / "raw"))
        if st.button("Inspect Dataset Schemas", key="btn_audit_dir"):
            try:
                res = audit_dataset(raw_dir)
                st.success(f"Found {res['file_count']} files.")
                for f in res["files"]:
                    with st.expander(f"{f['name']} ({f.get('size_bytes', 0):,} bytes)"):
                        st.caption(f"SHA-256: {f.get('sha256', '-')}")
                        if "table" in f:
                            st.write(f"Columns: {', '.join(f['table']['columns'])}")
                            st.json(f['table']['sample_values'])
            except Exception as exc:
                st.error(str(exc))

# -----------------------------------------------------------------------------
# TAB 6: Methodology & Research
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("Scientific Methodology and Ethical Safeguards")
    st.markdown(
        r"""
### 1. Research Scope and Dataset Provenance
- **Dataset**: Kenyan On-Station Maize Field Experiment (*Mutuku et al.*)
- **Source**: [Mendeley Data, Version 2](https://data.mendeley.com/datasets/8bx88zpvdp/2) | **License**: CC BY 4.0
- **Field Sites**: Chuka (Humic Nitisols) and Kandara (Humic Nitisols), Kenya.
- **Experimental Design**: Split-plot Randomized Complete Block Design (RCBD) across 4 cropping seasons (`LR16`, `LR17`, `SR16`, `SR17`).
- **Treatments**: 2 tillage depths (D0: Zero tillage, D15: Conventional 15cm) x 7 soil input regimes x 4 field replicates = 448 plot-season observations.

---

### 2. Binary Target Formulation and Proxy Disclosure
- **Target `high_yield`**: Defined strictly as `Yield_Mg/ha > 0.735 Mg/ha` (the empirical median across all 448 observations).
- **Class Balance**: Exactly 50.0% positive (224 plots) and 50.0% negative (224 plots).
- **Ethical and Agronomic Safeguard**: This classification label represents a relative within-dataset yield distinction. It is an explicit research proxy and **must not be represented as an autonomous irrigation hardware control signal** without rigorous on-farm sensor calibration.

---

### 3. Quantum Machine Learning Architecture
- **Simulator**: Local `qiskit.quantum_info.Statevector` simulator (exact classical simulation without sampling shot noise).
- **Feature Encoding**: Angle encoding mapping normalized feature values $x_i \in [-1, 1]$ into rotation angles $\theta_i = \pi x_i$ via $RY(\theta_i)$.
- **Variational Circuit (VQC)**: Alternating parameterized $RY(\theta) / RZ(\theta)$ single-qubit rotations with ring CNOT entanglement.
- **Quantum Kernel (QSVM)**: Directly evaluates Hilbert-space quantum state fidelity $K(x_i, x_j) = |\\langle\\phi(x_i)|\\phi(x_j)\\rangle|^2$ for dual quadratic optimization.
- **Optimization**: Gradient-free COBYLA optimizer minimizing regularized cross-entropy loss.

---

### 4. Hackathon Scoring and AI Policy
- Developed for **Qiskit Fall Fest Nairobi 2026 Hackathon (Challenge 08 + All Stretch Goals)**.
- Adheres to the official [AI Usage Policy](https://github.com/quantumke-org/Qiskit-Fall-Fest-Nairobi-2026-Hackathon-Prompts/blob/main/AI_POLICY.md).
        """
    )
