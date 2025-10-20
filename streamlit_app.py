import os, json, io, gzip, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
import shap

st.set_page_config(page_title="Pathogenicity Demo", layout="wide")

MODEL_PATH = "model_extratrees_ds4.gz"     # <- compressed file
FEATS_PATH = "feature_names.json"

@st.cache_resource
def load_artifacts():
    with gzip.open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    with open(FEATS_PATH) as f:
        feats = json.load(f)
    return model, feats

model, FEATURES = load_artifacts()

def clean_headers(df):
    df.columns = (df.columns.astype(str).str.replace(r'[\u200b\ufeff]','',regex=True).str.strip())
    return df

def read_table(file):
    if file.name.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(file)
    else:
        try:
            df = pd.read_csv(file, engine="python", sep=None, dtype=str,
                             na_values=["","NA","N/A"], quoting=3,
                             on_bad_lines="skip", low_memory=False, encoding_errors="ignore")
        except Exception:
            df = pd.read_csv(file, engine="python", sep=",", dtype=str,
                             na_values=["","NA","N/A"], quoting=3,
                             on_bad_lines="skip", low_memory=False, encoding_errors="ignore")
    return clean_headers(df)

def to_numeric(df):
    for c in df.columns:
        if df[c].dtype.kind in "OUS":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def prepare(df):
    for c in FEATURES:
        if c not in df.columns: df[c] = np.nan
    return to_numeric(df[FEATURES].copy())

st.title("Variant Pathogenicity Demo — ExtraTrees (Dataset-4)")
uploaded = st.file_uploader("Upload CSV/XLSX with trained feature columns", type=["csv","xlsx","xls"])
c1, c2 = st.columns(2)
with c1: run_pred = st.button("Predict")
with c2: run_explain = st.button("Explain first row")

if uploaded and (run_pred or run_explain):
    df_raw = read_table(uploaded)
    missing = [c for c in FEATURES if c not in df_raw.columns]
    if missing:
        st.error(f"Missing {len(missing)} required feature(s). First 10: {missing[:10]}")
    else:
        X = prepare(df_raw)
        if run_pred:
            proba = model.predict_proba(X)[:,1]
            pred = (proba > 0.5).astype(int)
            out = pd.DataFrame({"row_index": df_raw.index,
                                "pred_prob_pathogenic": proba,
                                "pred_label": pred})
            st.dataframe(out, use_container_width=True)
        if run_explain and len(X)>0:
            import matplotlib.pyplot as plt
            X_imp = pd.DataFrame(model.named_steps["imputer"].transform(X), columns=FEATURES)
            clf = model.named_steps["clf"]
            bg = X_imp.sample(min(200, len(X_imp)), random_state=0)
            expl = shap.Explainer(clf.predict_proba, bg, feature_names=FEATURES, algorithm="permutation")
            ex = expl(X_imp.iloc[[0]])
            vals = np.array(ex.values)
            vals_1d = vals[0, :, 1] if vals.ndim==3 else vals[0]
            s = pd.Series(vals_1d, index=FEATURES).sort_values(key=np.abs, ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(7,5))
            s.iloc[::-1].plot(kind="barh", ax=ax); ax.set_title("SHAP values (row #1)")
            fig.tight_layout(); st.pyplot(fig)
