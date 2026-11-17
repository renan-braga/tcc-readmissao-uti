"""
Utilitários de inferência — carrega modelos v7 e processa dados de paciente.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

MODELS_DIR    = "/Users/renanbraga/tcc_ccai/scripts/resultados/baseline_v7_per_snapshot"
SNAPSHOTS_DIR = "/Users/renanbraga/tcc_ccai/scripts"

SNAPSHOT_NAMES = ["D0", "D1", "D2", "D3"]
SNAPSHOT_LABELS = {
    "D0": "D0 — Admissão (0–24h)",
    "D1": "D1 — Estabilização (0–48h)",
    "D2": "D2 — Evolução (0–72h)",
    "D3": "D3 — Consolidação (0–96h)",
}

ID_COLS = [
    "stay_id", "subject_id", "hadm_id",
    "intime", "outtime", "admittime", "dischtime",
    "deathtime", "next_icu_intime",
    "readmitted", "hours_until_readmission", "discharge_location",
    "anchor_year", "anchor_year_group", "stay_rank",
]
SOFA_COMPONENTS = [
    "sofa_coagulation", "sofa_liver", "sofa_renal",
    "sofa_neurologic", "sofa_cardiovascular", "sofa_respiratory",
]
DELTA_COLS = [
    "sofa_pt", "sofa_pt_respiration", "sofa_pt_coagulation", "sofa_pt_liver",
    "sofa_pt_cardiovascular", "sofa_pt_neurologic", "sofa_pt_renal",
    "gcs_pt", "creatinine_pt", "platelet_pt", "bilirubin_pt",
    "meanbp_pt", "urine_output_pt", "pao2fio2_pt",
    "sofa_total", "creatinine_max", "creatinine_mean", "bilirubin_max",
    "map_mean", "heart_rate_mean", "gcs_total_min", "urine_output_ml",
    "po2_min", "bicarbonate_mean", "wbc_max", "norepinephrine_max",
    "any_vasopressor_max", "spo2_min", "respiratory_rate_mean",
    "systolic_bp_mean", "potassium_max",
]

# ── Pré-processamento (espelho do pipeline de treino) ─────────────────────────

def compute_deltas(curr_df, prev_df):
    cols = [c for c in DELTA_COLS if c in curr_df.columns and c in prev_df.columns]
    prev_sub = prev_df[["stay_id"] + cols].rename(columns={c: f"_prev_{c}" for c in cols})
    merged = curr_df.merge(prev_sub, on="stay_id", how="left")
    for c in cols:
        merged[f"delta_{c}"] = merged[c] - merged[f"_prev_{c}"]
    return merged.drop(columns=[f"_prev_{c}" for c in cols])


def engineer_features(df):
    df = df.copy()
    if "heart_rate_mean" in df.columns and "systolic_bp_mean" in df.columns:
        df["shock_index"]        = df["heart_rate_mean"] / df["systolic_bp_mean"].clip(lower=1)
        df["severe_hypotension"] = (df["systolic_bp_mean"] < 90).astype(int)
        df["tachycardia"]        = (df["heart_rate_mean"] > 100).astype(int)
    if "po2_min" in df.columns and "fio2_max" in df.columns:
        fio2 = df["fio2_max"].where(df["fio2_max"] > 1, df["fio2_max"] * 100)
        df["pf_ratio"]            = df["po2_min"] / fio2.clip(lower=21) * 100
        df["respiratory_failure"] = (df["pf_ratio"] < 200).astype(int)
    if "creatinine_max" in df.columns:
        df["aki_stage1"] = ((df["creatinine_max"] >= 1.5) & (df["creatinine_max"] < 2.0)).astype(int)
        df["aki_stage2"] = ((df["creatinine_max"] >= 2.0) & (df["creatinine_max"] < 3.0)).astype(int)
        df["aki_stage3"] = (df["creatinine_max"] >= 3.0).astype(int)
    if "gcs_total_min" in df.columns:
        df["gcs_severe"]   = (df["gcs_total_min"] <= 8).astype(int)
        df["gcs_moderate"] = ((df["gcs_total_min"] > 8) & (df["gcs_total_min"] <= 12)).astype(int)
    if "age" in df.columns:
        df["age_elderly"]      = (df["age"] >= 65).astype(int)
        df["age_very_elderly"] = (df["age"] >= 80).astype(int)
    if "icu_los_hours" in df.columns:
        df["prolonged_stay"] = (df["icu_los_hours"] > 168).astype(int)
    return df


def prepare_X_for_inference(df, bundle):
    """Aplica o mesmo pré-processamento do treino usando encoders+scaler salvos."""
    from sklearn.preprocessing import LabelEncoder
    encoders = bundle["encoders"]
    features  = bundle["features"]

    drop = [c for c in ID_COLS + SOFA_COMPONENTS if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    X = engineer_features(X)

    # Encode categoricals
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].fillna("UNKNOWN").astype(str)
        le = encoders.get(col)
        if le:
            known = set(le.classes_)
            X[col] = X[col].apply(lambda x: int(le.transform([x])[0]) if x in known else -1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Alinhar colunas com o modelo
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    return X

# ── Carregamento de modelos ───────────────────────────────────────────────────

def load_bundle(snapshot_name):
    path = os.path.join(MODELS_DIR, f"models_{snapshot_name.lower()}.joblib")
    return joblib.load(path)


def load_all_snapshots():
    snaps = {}
    prev = None
    for name in SNAPSHOT_NAMES:
        path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{name.lower()}.csv")
        df = pd.read_csv(path)
        if prev is not None:
            df = compute_deltas(df, prev)
        snaps[name] = df
        prev = pd.read_csv(os.path.join(SNAPSHOTS_DIR, f"snapshot_{name.lower()}.csv"))
    return snaps

# ── Busca por stay_id ─────────────────────────────────────────────────────────

def get_patient_snapshots(stay_id, all_snaps):
    """Retorna dict {snapshot_name: row_df} para um stay_id."""
    result = {}
    for name, df in all_snaps.items():
        row = df[df["stay_id"] == stay_id]
        if not row.empty:
            result[name] = row.copy()
    return result

# ── Inferência ────────────────────────────────────────────────────────────────

def predict_patient(patient_snaps, model_name="XGBoost"):
    """
    Retorna dict {snapshot: {"prob": float, "shap_values": array, "features": array, "feature_names": list}}
    """
    results = {}
    for snap_name, row_df in patient_snaps.items():
        bundle = load_bundle(snap_name)
        model  = bundle["models"][model_name]
        X = prepare_X_for_inference(row_df, bundle)

        prob = float(model.predict_proba(X)[0, 1])

        # SHAP
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            results[snap_name] = {
                "prob":          prob,
                "shap_values":   shap_values[0],
                "base_value":    float(explainer.expected_value
                                       if not isinstance(explainer.expected_value, list)
                                       else explainer.expected_value[1]),
                "feature_names": X.columns.tolist(),
                "feature_values": X.values[0],
            }
        except Exception:
            results[snap_name] = {
                "prob":          prob,
                "shap_values":   None,
                "base_value":    None,
                "feature_names": X.columns.tolist(),
                "feature_values": X.values[0],
            }
    return results

# ── Template CSV ──────────────────────────────────────────────────────────────

def get_template_csv(snapshot_name="D0"):
    bundle = load_bundle(snapshot_name)
    features = bundle["features"]
    template = pd.DataFrame(columns=["stay_id", "subject_id"] + features)
    template.loc[0] = ["", ""] + [np.nan] * len(features)
    return template


