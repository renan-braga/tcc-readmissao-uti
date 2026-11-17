"""
Tuning de Hiperparâmetros — LightGBM + XGBoost via Optuna
MIMIC-IV v3.1

Busca bayesiana (Optuna) com 5-fold cross-validation patient-level.
Otimiza AUC-ROC no snapshot D1 (melhor sinal × dados suficientes).
Aplica os melhores parâmetros em todos os snapshots (D0–D3) e salva
os resultados em baseline_v6_tuned/.

Autor: Renan de Almeida Simões Braga
"""

import os
import warnings
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.switch_backend("Agg")
sns.set_style("whitegrid")

SCRIPTS_DIR  = "/Users/renanbraga/tcc_ccai/scripts/features_brutas"
SNAPSHOTS_DIR= "/Users/renanbraga/tcc_ccai/scripts"
RESULTS_DIR  = "/Users/renanbraga/tcc_ccai/scripts/resultados/baseline_v6_tuned"
N_TRIALS     = 80
N_FOLDS      = 5
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET_RECALL= 0.60
TARGET_AUC   = 0.75

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
SOFA_COMPONENTS = [
    "sofa_coagulation", "sofa_liver", "sofa_renal",
    "sofa_neurologic", "sofa_cardiovascular", "sofa_respiratory",
]
DELTA_COLS = [
    "sofa_pt",
    "sofa_pt_respiration", "sofa_pt_coagulation", "sofa_pt_liver",
    "sofa_pt_cardiovascular", "sofa_pt_neurologic", "sofa_pt_renal",
    "gcs_pt", "creatinine_pt", "platelet_pt", "bilirubin_pt",
    "meanbp_pt", "urine_output_pt", "pao2fio2_pt",
    "sofa_total",
    "creatinine_max", "creatinine_mean", "bilirubin_max",
    "map_mean", "heart_rate_mean", "gcs_total_min", "urine_output_ml",
    "po2_min", "bicarbonate_mean", "wbc_max", "norepinephrine_max",
    "any_vasopressor_max", "spo2_min", "respiratory_rate_mean",
    "systolic_bp_mean", "potassium_max",
]

# ── Funções auxiliares (espelho do script 03) ─────────────────────────────────

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
    drop = [c for c in ID_COLS + SOFA_COMPONENTS if c in df.columns]
    X = df.drop(columns=drop)
    X = engineer_features(X)
    X, encoders = encode_categoricals(X, encoders)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if scaler is not None:
        if fit:
            X_s = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        else:
            X_s = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        return X, X_s, encoders, scaler
    return X, encoders


def patient_level_split(df):
    patient_label = df.groupby("subject_id")["readmitted"].max()
    train_pts, test_pts = train_test_split(
        patient_label.index, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=patient_label.values,
    )
    return df[df["subject_id"].isin(train_pts)].copy(), df[df["subject_id"].isin(test_pts)].copy()

# ── Cross-validation patient-level ───────────────────────────────────────────

def patient_level_cv(df, n_splits=N_FOLDS):
    """Yields (train_idx, val_idx) garantindo que stays do mesmo paciente
    ficam no mesmo fold. Estratificado por 'paciente teve readmissão'."""
    patient_label = df.groupby("subject_id")["readmitted"].max().reset_index()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for train_pts_pos, val_pts_pos in skf.split(patient_label["subject_id"],
                                                  patient_label["readmitted"]):
        train_subjects = set(patient_label.iloc[train_pts_pos]["subject_id"])
        val_subjects   = set(patient_label.iloc[val_pts_pos]["subject_id"])
        train_idx = df.index[df["subject_id"].isin(train_subjects)].tolist()
        val_idx   = df.index[df["subject_id"].isin(val_subjects)].tolist()
        yield train_idx, val_idx


def cv_auc(df_train, params, model_type="lgb"):
    """Retorna AUC médio de N_FOLDS folds com patient-level split."""
    aucs = []
    for train_idx, val_idx in patient_level_cv(df_train):
        fold_train = df_train.loc[train_idx]
        fold_val   = df_train.loc[val_idx]

        X_tr, enc = prepare_X(fold_train)
        y_tr = fold_train["readmitted"].values

        X_val, _ = prepare_X(fold_val, encoders=enc)
        for col in X_tr.columns:
            if col not in X_val.columns:
                X_val[col] = 0
        X_val = X_val[X_tr.columns]
        y_val = fold_val["readmitted"].values

        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        if model_type == "lgb":
            model = lgb.LGBMClassifier(**params, scale_pos_weight=pos_weight,
                                        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        else:
            model = xgb.XGBClassifier(**params, scale_pos_weight=pos_weight,
                                       random_state=RANDOM_STATE, n_jobs=-1,
                                       eval_metric="auc", verbosity=0)

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_prob))

    return float(np.mean(aucs)), float(np.std(aucs))

# ── Objectives Optuna ─────────────────────────────────────────────────────────

def make_lgb_objective(df_train):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 5.0),
        }
        mean_auc, _ = cv_auc(df_train, params, model_type="lgb")
        return mean_auc
    return objective


def make_xgb_objective(df_train):
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 600),
            "max_depth":       trial.suggest_int("max_depth", 3, 7),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":trial.suggest_int("min_child_weight", 3, 30),
            "gamma":           trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0.1, 5.0),
        }
        mean_auc, _ = cv_auc(df_train, params, model_type="xgb")
        return mean_auc
    return objective

# ── Avaliação final num snapshot ──────────────────────────────────────────────

def compute_metrics(model, X_test, y_test, X_train=None, y_train=None):
    y_prob = model.predict_proba(X_test)[:, 1]
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, y_prob)
    mask = rec_arr[:-1] >= TARGET_RECALL
    threshold = float(thr_arr[mask][np.argmax(prec_arr[:-1][mask])]) if mask.any() else 0.3
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    result = {
        "auc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "brier": brier_score_loss(y_test, y_prob),
        "recall": report["1"]["recall"],
        "precision": report["1"]["precision"],
        "f1": report["1"]["f1-score"],
        "threshold": threshold,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
    if X_train is not None and y_train is not None:
        result["auc_train"]   = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        result["overfitting"] = result["auc_train"] - result["auc"]
    return result


def train_snapshot_tuned(snapshot_name, filepath, best_params, prev_df=None):
    print(f"\n{'='*60}")
    print(f"  Snapshot {snapshot_name}  [tuned]")
    print(f"{'='*60}")

    df = pd.read_csv(filepath)
    raw_df = df.copy()

    if prev_df is not None:
        df = compute_deltas(df, prev_df)
        n_d = sum(1 for c in df.columns if c.startswith("delta_"))
        print(f"  Delta features: {n_d}")

    print(f"  {df.shape[0]:,} linhas | {df['readmitted'].mean():.3f} taxa readmissão")
    train_df, test_df = patient_level_split(df)
    print(f"  Treino: {len(train_df):,} | Teste: {len(test_df):,}")

    y_train = train_df["readmitted"].values
    y_test  = test_df["readmitted"].values
    pw      = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

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

    print("\n  Treinando modelos (params tuned para LGB e XGB)...")
    models = {
        "LogReg": LogisticRegression(class_weight="balanced", max_iter=1000,
                                     random_state=42, C=0.1).fit(X_train_s, y_train),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=6,
                                               min_samples_leaf=10, class_weight="balanced",
                                               random_state=42, n_jobs=-1).fit(X_train, y_train),
        "XGBoost":  xgb.XGBClassifier(**best_params["xgb"], scale_pos_weight=pw,
                                        random_state=42, n_jobs=-1,
                                        eval_metric="auc", verbosity=0).fit(X_train, y_train),
        "LightGBM": lgb.LGBMClassifier(**best_params["lgb"], scale_pos_weight=pw,
                                         random_state=42, n_jobs=-1, verbose=-1).fit(X_train, y_train),
    }

    print("\n  Avaliando...")
    results = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "LogReg" else X_train
        X_te = X_test_s  if name == "LogReg" else X_test_raw
        results[name] = compute_metrics(model, X_te, y_test, X_tr, y_train)

    print(f"\n  {'Modelo':<14} {'AUC':>6} {'AUPRC':>6} {'Brier':>6} {'Recall':>7} {'Prec':>6} {'F1':>6} {'Overfit':>8}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        ov = f"{r.get('overfitting',0):+.3f}"
        ok = "✅" if r["auc"] >= TARGET_AUC and r["recall"] >= TARGET_RECALL else "❌"
        print(f"  {name:<14} {r['auc']:>6.3f} {r['auprc']:>6.3f} {r['brier']:>6.3f} "
              f"{r['recall']:>7.3f} {r['precision']:>6.3f} {r['f1']:>6.3f} {ov:>8} {ok}")

    # ROC/PR
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} ({res['auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec, prec, lw=2, label=f"{name} ({res['auprc']:.3f})")
    axes[0].plot([0,1],[0,1],"k--",alpha=0.4)
    axes[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC — {snapshot_name} [tuned]")
    axes[0].legend(fontsize=9)
    axes[1].axhline(y_test.mean(), color="gray", ls="--", alpha=0.5)
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"PR — {snapshot_name} [tuned]")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"roc_comparison_{snapshot_name}.png"),
                dpi=150, bbox_inches="tight")
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
        })
    return rows, raw_df

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("TUNING — LightGBM + XGBoost via Optuna")
    print(f"Snapshot de tuning: D1 | Trials: {N_TRIALS} | CV: {N_FOLDS}-fold patient-level")
    print("=" * 60)

    # Carrega D0 (para deltas) e D1 (para tuning)
    d0_path = os.path.join(SNAPSHOTS_DIR, "snapshot_d0.csv")
    d1_path = os.path.join(SNAPSHOTS_DIR, "snapshot_d1.csv")
    if not os.path.exists(d1_path):
        print("snapshot_d1.csv não encontrado. Rode 02_snapshots_temporais.py primeiro.")
        return

    df_d0 = pd.read_csv(d0_path)
    df_d1 = pd.read_csv(d1_path)
    df_d1 = compute_deltas(df_d1, df_d0)

    # Usa só o conjunto de treino para tuning (evita contaminação do teste)
    train_df, _ = patient_level_split(df_d1)
    print(f"\nDados para tuning (treino D1): {len(train_df):,} stays")
    print(f"Positivos: {train_df['readmitted'].mean():.3f} ({train_df['readmitted'].mean()*100:.1f}%)\n")

    best_params = {}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f"Otimizando LightGBM ({N_TRIALS} trials)...")
    study_lgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(make_lgb_objective(train_df), n_trials=N_TRIALS,
                       show_progress_bar=True)
    best_lgb = study_lgb.best_params
    best_lgb_auc = study_lgb.best_value
    best_params["lgb"] = best_lgb
    print(f"\n  Melhor AUC CV (LGB): {best_lgb_auc:.4f}")
    print(f"  Params: {best_lgb}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f"\nOtimizando XGBoost ({N_TRIALS} trials)...")
    study_xgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(make_xgb_objective(train_df), n_trials=N_TRIALS,
                       show_progress_bar=True)
    best_xgb = study_xgb.best_params
    best_xgb_auc = study_xgb.best_value
    best_params["xgb"] = best_xgb
    print(f"\n  Melhor AUC CV (XGB): {best_xgb_auc:.4f}")
    print(f"  Params: {best_xgb}")

    # Salva params
    params_path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(params_path, "w") as f:
        json.dump({
            "lgb": {**best_lgb, "cv_auc": round(best_lgb_auc, 4)},
            "xgb": {**best_xgb, "cv_auc": round(best_xgb_auc, 4)},
        }, f, indent=2)
    print(f"\nParâmetros salvos: {params_path}")

    # ── Aplica em todos os snapshots ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AVALIAÇÃO FINAL — Todos os snapshots com params tuned")
    print("=" * 60)

    snapshots = [
        (sid, os.path.join(SNAPSHOTS_DIR, f"snapshot_{sid.lower()}.csv"))
        for sid in ["D0", "D1", "D2", "D3"]
        if os.path.exists(os.path.join(SNAPSHOTS_DIR, f"snapshot_{sid.lower()}.csv"))
    ]

    all_rows = []
    prev_df  = None
    for name, path in snapshots:
        try:
            rows, prev_df = train_snapshot_tuned(name, path, best_params, prev_df)
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
    print("RESUMO FINAL [tuned]")
    print("=" * 60)
    pivot = summary.pivot_table(index="model", columns="snapshot",
                                values="auc", aggfunc="first").round(3)
    print(pivot.to_string())
    print(f"\nResumo salvo: {out}")


if __name__ == "__main__":
    main()
