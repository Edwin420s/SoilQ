from pathlib import Path
import json
import pandas as pd
import streamlit as st
from soilq.data import audit_dataset
from soilq.config import PROJECT_ROOT

st.set_page_config(page_title="SoilQ Research Dashboard", page_icon="🌱", layout="wide")
st.title("🌱 SoilQ — Research Dashboard")
st.caption("Dataset audit and model comparison. Not a farm-control or irrigation-advice system.")
tab1, tab2, tab3 = st.tabs(["Dataset audit", "Model report", "Methodology"])
with tab1:
    path = st.text_input("Local file or directory", str(PROJECT_ROOT / "data" / "raw"))
    if st.button("Audit dataset"):
        try: st.json(audit_dataset(path))
        except Exception as exc: st.error(str(exc))
    st.info("Place the downloaded Kenyan maize dataset in data/raw and audit it before choosing a target.")
with tab2:
    report_path = st.text_input("Report JSON", str(PROJECT_ROOT / "data" / "reports" / "experiment.json"))
    if st.button("Load report"):
        try:
            r = json.loads(Path(report_path).read_text(encoding="utf-8"))
            st.json(r)
            models = r.get("models", {})
            if models:
                frame = pd.DataFrame([{"Model": k, **{m:v.get(m) for m in ("accuracy","balanced_accuracy","precision","recall","f1","roc_auc","fit_seconds")}} for k,v in models.items()])
                st.dataframe(frame, use_container_width=True)
                st.bar_chart(frame.set_index("Model")[["f1"]])
        except Exception as exc: st.error(f"Could not load report: {exc}")
with tab3:
    st.markdown("""
**Workflow:** audit files → define target from evidence → select numeric features → benchmark models → inspect limitations.

The Qiskit model runs on a local Statevector simulator. It does not prove quantum advantage and must not be used for real irrigation decisions without appropriate field validation.
""")
