# --- bootstrap: force-install deps if Hugging Face didn't ---
import sys, subprocess
def _need(modname):
    try: __import__(modname); return False
    except ImportError: return True
def _pip(pkg_spec):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec])

PKGS = [
    ("joblib",        "1.4.2", "joblib"),
    ("numpy",         "1.26.4","numpy"),
    ("pandas",        "2.2.2", "pandas"),
    ("scikit-learn",  "1.4.2", "sklearn"),
    ("scipy",         "1.11.4","scipy"),
    ("shap",          "0.46.0","shap"),
    ("matplotlib",    "3.8.4", "matplotlib"),
    ("openpyxl",      "3.1.5", "openpyxl"),
    ("pillow",        None,    "PIL"),
    ("gradio",        "4.44.1","gradio"),
]
for pkg, ver, mod in PKGS:
    if _need(mod):
        _pip(f"{pkg}=={ver}" if ver else pkg)

# now safe to import everything
import os, json, io, gzip, joblib, traceback
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from PIL import Image
import shap
import gradio as gr

# ---- artifacts in repo root ----
MODEL_PATH  = "model_extratrees_ds4.joblib"   # or "model_extratrees_ds4.gz"
FEATS_PATH  = "feature_names.json"

# if you uploaded a .gz model, uncomment these two lines and delete the next joblib.load:
# with gzip.open(MODEL_PATH, "rb") as f:
#     model = joblib.load(f)
model = joblib.load(MODEL_PATH)

with open(FEATS_PATH) as f:
    FEATURES = json.load(f)

# ---------- helpers ----------
def _clean_headers(df):
    df.columns = (df.columns.astype(str)
                  .str.replace(r'[\u200b\ufeff]', '', regex=True)
                  .str.strip())
    return df

def _read_table(file):
    name = getattr(file, "name", str(file))
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(name, engine="openpyxl")
    else:
        try:
            df = pd.read_csv(name, engine="python", sep=None, dtype=str,
                             na_values=["", "NA", "N/A"], quoting=3,
                             on_bad_lines="skip", low_memory=False,
                             encoding_errors="ignore")
        except Exception:
            df = None
            for sep in [",", "\t", ";", "|"]:
                try:
                    df = pd.read_csv(name, engine="python", sep=sep, dtype=str,
                                     na_values=["", "NA", "N/A"], quoting=3,
                                     on_bad_lines="skip", low_memory=False,
                                     encoding_errors="ignore")
                    break
                except Exception:
                    pass
            if df is None:
                raise
    return _clean_headers(df)

def _coerce_numeric(df):
    for c in df.columns:
        if df[c].dtype.kind in "OUS":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _prepare(df):
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    return _coerce_numeric(df[FEATURES].copy())

def _validate_columns(df):
    missing = [c for c in FEATURES if c not in df.columns]
    extra = [c for c in df.columns if c not in FEATURES]
    return missing, extra

# ---------- app fns ----------
def predict_csv(file):
    try:
        if file is None: return None, "No file."
        df = _read_table(file)
        missing, _ = _validate_columns(df)
        if missing: return None, f"Missing {len(missing)} feature(s). First 10: {missing[:10]}"
        X = _prepare(df)
        proba = model.predict_proba(X)[:, 1]
        pred = (proba > 0.5).astype(int)
        out = pd.DataFrame({"row_index": df.index,
                            "pred_prob_pathogenic": proba,
                            "pred_label": pred})
        return out, "OK"
    except Exception:
        return None, "ERROR:\n" + traceback.format_exc()

def explain_first_row(file):
    try:
        if file is None: return None, "No file."
        df = _read_table(file)
        missing, _ = _validate_columns(df)
        if missing: return None, f"Missing features for SHAP. First 10: {missing[:10]}"
        X = _prepare(df)
        if X.shape[0] == 0: return None, "No rows."
        X_imp = pd.DataFrame(model.named_steps["imputer"].transform(X), columns=FEATURES)
        clf = model.named_steps["clf"]

        bg = X_imp.sample(min(200, len(X_imp)), random_state=0)
        expl = shap.Explainer(clf.predict_proba, bg, feature_names=FEATURES, algorithm="permutation")
        ex = expl(X_imp.iloc[[0]])
        vals = np.array(ex.values)
        vals_1d = vals[0, :, 1] if vals.ndim == 3 else vals[0]
        s = pd.Series(vals_1d, index=FEATURES).sort_values(key=np.abs, ascending=False).head(15)

        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        s.iloc[::-1].plot(kind="barh", ax=ax)
        ax.set_title("SHAP values (row #1)")
        fig.tight_layout()
        return fig, "OK"
    except Exception:
        return None, "ERROR:\n" + traceback.format_exc()

# ---------- UI ----------
with gr.Blocks(title="Variant Pathogenicity Demo — ExtraTrees (Dataset-4)") as demo:
    gr.Markdown("### Variant Pathogenicity Demo — ExtraTrees (Dataset-4)")
    inp = gr.File(label="Upload CSV/XLSX containing the trained feature columns")
    with gr.Row():
        btn_pred = gr.Button("Predict")
        btn_expl = gr.Button("Explain first row")
    out_df = gr.Dataframe(label="Predictions")
    out_plot = gr.Plot(label="SHAP (row #1)")
    out_txt = gr.Markdown()
    btn_pred.click(fn=predict_csv, inputs=inp, outputs=[out_df, out_txt])
    btn_expl.click(fn=explain_first_row, inputs=inp, outputs=[out_plot, out_txt])

if __name__ == "__main__":
    port_env = os.environ.get("GRADIO_SERVER_PORT", "")
    port = int(port_env) if port_env.isdigit() else None
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, debug=True)
