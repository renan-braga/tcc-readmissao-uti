"""
Treinamento — Abordagem Lean por Scores Clínicos
MIMIC-IV v3.1

Usa apenas scores validados (OASIS, CCI, APSIII, SAPSII, SOFA pt, KDIGO)
e features de contexto. ~27 features em D0, ~41 em D1–D3 com deltas.
Objetivo: reduzir overfitting de D2/D3 com maior razão amostras/feature.

Autor: Renan de Almeida Simões Braga
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")
sns.set_style("whitegrid")

SCRIPTS_DIR  = "/Users/renanbraga/tcc_ccai/scripts/scores_lean"
RESULTS_DIR  = "/Users/renanbraga/tcc_ccai/scripts/resultados/baseline_v5_scores_lean"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
TARGET_RECALL = 0.60
TARGET_AUC    = 0.75

ID_COLS = [
    "stay_id", "subject_id", "hadm_id",
    "intime", "outtime", "admittime", "dischtime",
    "deathtime", "next_icu_intime",
    "readmitted", "hours_until_readmission", "discharge_location",
    "anchor_year", "anchor_year_group", "stay_rank",
    # Removidas em set/2026 — ver docs/TODO.md seção 20:
    "icu_los_hours",   # é o LOS TOTAL da estadia: no corte de 24h isso é informação do futuro
    "last_careunit",   # idêntica a first_careunit em 100% das estadias (MIMIC separa transferências
                       # entre UTIs em estadias distintas); duplicata fragmenta a atribuição SHAP
]

# Colunas para delta entre snapshots consecutivos
# Apenas as point-in-time fazem sentido como delta real
DELTA_COLS = [
    "sofa_pt",
    "sofa_pt_respiration", "sofa_pt_coagulation", "sofa_pt_liver",
    "sofa_pt_cardiovascular", "sofa_pt_neurologic", "sofa_pt_renal",
    "gcs_pt", "creatinine_pt", "platelet_pt", "bilirubin_pt",
    "meanbp_pt", "urine_output_pt", "pao2fio2_pt",
    "kdigo_stage",
]

# ── Delta entre snapshots ──────────────────────────────────────────────────────

def compute_deltas(curr_df, prev_df):
    cols = [c for c in DELTA_COLS if c in curr_df.columns and c in prev_df.columns]
    prev_sub = prev_df[["stay_id"] + cols].rename(columns={c: f"_prev_{c}" for c in cols})
    merged = curr_df.merge(prev_sub, on="stay_id", how="left")
    for c in cols:
        merged[f"delta_{c}"] = merged[c] - merged[f"_prev_{c}"]
    return merged.drop(columns=[f"_prev_{c}" for c in cols])

# ── Preparação de features ────────────────────────────────────────────────────

def encode_categoricals(df, encoders=None):
    fit = encoders is None
    if fit:
        encoders = {}
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("UNKNOWN").astype(str)
        if fit:
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: int(le.transform([x])[0]) if x in known else -1)
    return df, encoders


def prepare_X(df, encoders=None, scaler=None, fit=True):
    drop = [c for c in ID_COLS if c in df.columns]
    X = df.drop(columns=drop)
    X, encoders = encode_categoricals(X, encoders)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if scaler is not None:
        if fit:
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        else:
            X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        return X, X_scaled, encoders, scaler
    return X, encoders


def patient_level_split(df):
    patient_label = df.groupby("subject_id")["readmitted"].max()
    train_patients, test_patients = train_test_split(
        patient_label.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=patient_label.values,
    )
    return df[df["subject_id"].isin(train_patients)].copy(), df[df["subject_id"].isin(test_patients)].copy()

# ── Modelos ───────────────────────────────────────────────────────────────────

def build_logreg(X, y):
    return LogisticRegression(class_weight="balanced", max_iter=1000,
                              random_state=42, solver="lbfgs", C=0.1).fit(X, y)

def build_rf(X, y):
    return RandomForestClassifier(n_estimators=200, max_depth=5,
                                  min_samples_leaf=10, class_weight="balanced",
                                  random_state=42, n_jobs=-1).fit(X, y)

def build_xgboost(X, y):
    pw = (y == 0).sum() / max((y == 1).sum(), 1)
    return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                              gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                              scale_pos_weight=pw, random_state=42, n_jobs=-1,
                              eval_metric="auc", verbosity=0).fit(X, y)

def build_lightgbm(X, y):
    pw = (y == 0).sum() / max((y == 1).sum(), 1)
    return lgb.LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
                               reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=pw,
                               random_state=42, n_jobs=-1, verbose=-1).fit(X, y)

# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test, X_train=None, y_train=None):
    y_prob = model.predict_proba(X_test)[:, 1]
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, y_prob)
    mask = rec_arr[:-1] >= TARGET_RECALL
    threshold = float(thr_arr[mask][np.argmax(prec_arr[:-1][mask])]) if mask.any() else 0.3
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    result = {
        "auc":       roc_auc_score(y_test, y_prob),
        "auprc":     average_precision_score(y_test, y_prob),
        "brier":     brier_score_loss(y_test, y_prob),
        "recall":    report["1"]["recall"],
        "precision": report["1"]["precision"],
        "f1":        report["1"]["f1-score"],
        "threshold": threshold,
        "y_prob":    y_prob,
        "y_pred":    y_pred,
    }
    if X_train is not None and y_train is not None:
        result["auc_train"]   = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        result["overfitting"] = result["auc_train"] - result["auc"]
    return result

# ── SHAP ──────────────────────────────────────────────────────────────────────

def run_shap(model, X_test, y_test, y_prob, model_name, snapshot_name, out_dir):
    print(f"    SHAP para {model_name}...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP — {snapshot_name} | {model_name} [scores lean]", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_beeswarm_{snapshot_name}_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return
    for idx, label in [(pos_idx[np.argmax(y_prob[pos_idx])], "readmitido"),
                       (neg_idx[np.argmin(y_prob[neg_idx])], "nao_readmitido")]:
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
        plt.title(f"SHAP — {label} | {snapshot_name} | {model_name} P={y_prob[idx]:.2f}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_waterfall_{snapshot_name}_{model_name}_{label}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    print(f"      Salvo: beeswarm + 2 waterfalls")

# ── Pipeline por snapshot ─────────────────────────────────────────────────────

def train_snapshot(snapshot_name, filepath, prev_df=None):
    print(f"\n{'='*60}")
    print(f"  Snapshot {snapshot_name}  [scores lean]")
    print(f"{'='*60}")

    df = pd.read_csv(filepath)
    raw_df = df.copy()

    if prev_df is not None:
        df = compute_deltas(df, prev_df)
        n_delta = sum(1 for c in df.columns if c.startswith("delta_"))
        print(f"  Features delta adicionadas: {n_delta}")

    print(f"  {df.shape[0]:,} linhas | {df['readmitted'].mean():.3f} taxa readmissão")
    train_df, test_df = patient_level_split(df)
    print(f"  Treino: {len(train_df):,} | Teste: {len(test_df):,}")

    y_train = train_df["readmitted"].values
    y_test  = test_df["readmitted"].values

    scaler = StandardScaler()
    X_train, X_train_s, enc, scaler = prepare_X(train_df, scaler=scaler, fit=True)
    X_test_raw, X_test_s, _, _      = prepare_X(test_df, encoders=enc, scaler=scaler, fit=False)

    for col in X_train.columns:
        if col not in X_test_raw.columns:
            X_test_raw[col] = 0
            X_test_s[col]   = 0
    X_test_raw = X_test_raw[X_train.columns]
    X_test_s   = X_test_s[X_train.columns]
    print(f"  Features: {X_train.shape[1]}")

    print("\n  Treinando modelos...")
    models = {}
    for name, builder, X in [
        ("LogReg",       build_logreg,    X_train_s),
        ("RandomForest", build_rf,        X_train),
        ("XGBoost",      build_xgboost,   X_train),
        ("LightGBM",     build_lightgbm,  X_train),
    ]:
        print(f"    {name}...", end=" ")
        models[name] = builder(X, y_train)
        print("ok")

    print("\n  Avaliando...")
    results = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "LogReg" else X_train
        X_te = X_test_s  if name == "LogReg" else X_test_raw
        results[name] = compute_metrics(model, X_te, y_test, X_tr, y_train)

    print(f"\n  {'Modelo':<14} {'AUC':>6} {'AUPRC':>6} {'Brier':>6} {'Recall':>7} {'Prec':>6} {'F1':>6} {'Overfit':>8}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        ov = f"{r.get('overfitting', 0):+.3f}"
        ok = "✅" if r["auc"] >= TARGET_AUC and r["recall"] >= TARGET_RECALL else "❌"
        print(f"  {name:<14} {r['auc']:>6.3f} {r['auprc']:>6.3f} {r['brier']:>6.3f} "
              f"{r['recall']:>7.3f} {r['precision']:>6.3f} {r['f1']:>6.3f} {ov:>8} {ok}")

    print("\n  Gerando SHAP...")
    for name in ["XGBoost", "LightGBM"]:
        try:
            run_shap(models[name], X_test_raw, y_test, results[name]["y_prob"],
                     name, snapshot_name, RESULTS_DIR)
        except Exception as e:
            print(f"    SHAP {name} falhou: {e}")

    # ROC/PR
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} ({res['auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec, prec, lw=2, label=f"{name} ({res['auprc']:.3f})")
    axes[0].plot([0,1],[0,1],"k--",alpha=0.4)
    axes[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC — {snapshot_name} [scores lean]")
    axes[0].legend(fontsize=9)
    axes[1].axhline(y_test.mean(), color="gray", ls="--", alpha=0.5)
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"PR — {snapshot_name} [scores lean]")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"roc_comparison_{snapshot_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({
        "models": models, "encoders": enc, "scaler": scaler,
        "features": X_train.columns.tolist(),
        "metrics": {n: {k: v for k, v in r.items() if k not in ("y_prob","y_pred")}
                    for n, r in results.items()},
    }, os.path.join(RESULTS_DIR, f"models_{snapshot_name.lower()}.joblib"))

    rows = []
    for name, r in results.items():
        rows.append({
            "snapshot": snapshot_name, "model": name,
            "train_n": len(train_df), "test_n": len(test_df),
            "auc": round(r["auc"],4), "auprc": round(r["auprc"],4),
            "brier": round(r["brier"],4), "recall": round(r["recall"],4),
            "precision": round(r["precision"],4), "f1": round(r["f1"],4),
            "overfitting": round(r.get("overfitting",0),4),
            "threshold": round(r["threshold"],3),
        })
    return rows, raw_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("TREINAMENTO SCORES LEAN — READMISSÃO UTI 72h — MIMIC-IV v3.1")
    print(f"Split: patient-level 80/20 (seed={RANDOM_STATE})")
    print(f"Modelos: LogReg | RandomForest | XGBoost | LightGBM")
    print("=" * 60)

    snapshots = [
        (sid, os.path.join(SCRIPTS_DIR, f"snapshot_{sid.lower()}.csv"))
        for sid in ["D0", "D1", "D2", "D3"]
        if os.path.exists(os.path.join(SCRIPTS_DIR, f"snapshot_{sid.lower()}.csv"))
    ]
    if not snapshots:
        print("Nenhum snapshot encontrado. Rode primeiro 02_snapshots_scores.py.")
        return

    print(f"Snapshots encontrados: {[s[0] for s in snapshots]}\n")

    all_rows = []
    prev_df  = None
    for name, path in snapshots:
        try:
            rows, prev_df = train_snapshot(name, path, prev_df=prev_df)
            all_rows.extend(rows)
        except Exception as e:
            import traceback
            print(f"ERRO no snapshot {name}: {e}")
            traceback.print_exc()
            prev_df = None

    summary = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "training_summary.csv")
    summary.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print("RESUMO FINAL")
    print("=" * 60)
    pivot = summary.pivot_table(index="model", columns="snapshot", values="auc", aggfunc="first").round(3)
    print(pivot.to_string())
    print(f"\nResumo salvo: {out}")


if __name__ == "__main__":
    main()
