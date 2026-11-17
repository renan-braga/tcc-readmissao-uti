"""
Modelo de Trajetória D0→D3 — linha adicional na tabela comparativa
MIMIC-IV v3.1

Concatena features de todos os snapshots (D0+D1+D2+D3) em um único vetor
por paciente e treina os mesmos modelos dos snapshots individuais.
Gera uma linha "D0→D3" para comparação direta na tabela de resultados.

Comparação justa: D3-baseline retreinado na mesma coorte (LOS ≥ 96h).

Autor: Renan de Almeida Simões Braga
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, classification_report,
)

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")
sns.set_style("whitegrid")

SNAPSHOTS_DIR = "/Users/renanbraga/tcc_ccai/scripts"
RESULTS_V7    = "/Users/renanbraga/tcc_ccai/scripts/resultados/baseline_v8_sem_vazamento"  # mesma versão (set/2026)
RESULTS_DIR   = "/Users/renanbraga/tcc_ccai/scripts/resultados/baseline_v8_sem_vazamento"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
TARGET_RECALL = 0.60

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

# ── Funções auxiliares ────────────────────────────────────────────────────────

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


def prepare_X(df, encoders=None, scaler=None, fit=True, skip_engineer=False):
    drop = [c for c in ID_COLS + SOFA_COMPONENTS if c in df.columns]
    X = df.drop(columns=drop)
    if not skip_engineer:
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


def ensemble_metrics(results, y_test, y_train, X_train, models):
    y_prob = (results["XGBoost"]["y_prob"] + results["LightGBM"]["y_prob"]) / 2
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, y_prob)
    mask = rec_arr[:-1] >= TARGET_RECALL
    threshold = float(thr_arr[mask][np.argmax(prec_arr[:-1][mask])]) if mask.any() else 0.3
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    y_prob_tr = (
        models["XGBoost"].predict_proba(X_train)[:, 1] +
        models["LightGBM"].predict_proba(X_train)[:, 1]
    ) / 2
    return {
        "auc":         roc_auc_score(y_test, y_prob),
        "auprc":       average_precision_score(y_test, y_prob),
        "brier":       brier_score_loss(y_test, y_prob),
        "recall":      report["1"]["recall"],
        "precision":   report["1"]["precision"],
        "f1":          report["1"]["f1-score"],
        "threshold":   threshold,
        "y_prob":      y_prob,
        "y_pred":      y_pred,
        "auc_train":   roc_auc_score(y_train, y_prob_tr),
        "overfitting": roc_auc_score(y_train, y_prob_tr) - roc_auc_score(y_test, y_prob),
    }

# ── Construção do dataset de trajetória ───────────────────────────────────────

def build_trajectory_dataset():
    """
    Carrega os 4 snapshots, computa deltas encadeados e faz inner join
    por stay_id. Cada coluna recebe prefixo d0_, d1_, d2_, d3_.

    Correções v2:
    - engineer_features aplicado ANTES do prefixo (senão prepare_X não encontra
      as colunas originais e as features derivadas ficam ausentes no D0→D3)
    - SOFA_COMPONENTS dropados ANTES do prefixo (consistência com prepare_X
      dos snapshots individuais, que dropa sofa_coagulation etc.)
    """
    print("Carregando snapshots...")
    snaps = {}
    prev = None
    for name in ["D0", "D1", "D2", "D3"]:
        path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{name.lower()}.csv")
        df = pd.read_csv(path)
        if prev is not None:
            df = compute_deltas(df, prev)
        # Fix 1: engenharia de features ANTES de prefixar
        feature_cols_now = [c for c in df.columns if c not in ID_COLS]
        df_feat = df[feature_cols_now + ["stay_id"]].copy()
        df_feat = engineer_features(df_feat)
        # Fix 2: drop SOFA_COMPONENTS ANTES de prefixar (mesma lógica do prepare_X)
        drop_sofa = [c for c in SOFA_COMPONENTS if c in df_feat.columns]
        df_feat = df_feat.drop(columns=drop_sofa)
        snaps[name] = {"feat": df_feat, "raw": df}
        prev = pd.read_csv(os.path.join(SNAPSHOTS_DIR, f"snapshot_{name.lower()}.csv"))
        print(f"  {name}: {df.shape[0]:,} stays, {df_feat.shape[1]} feature cols (após eng+drop_sofa)")

    # Colunas de identidade/meta — manter apenas do D3 (mais completo)
    meta_cols = [c for c in ID_COLS if c in snaps["D3"]["raw"].columns]

    # Prefixar features de cada snapshot
    prefixed = []
    for name, snap in snaps.items():
        prefix = name.lower() + "_"
        df_feat = snap["feat"]
        feature_cols = [c for c in df_feat.columns if c != "stay_id"]
        renamed = df_feat[["stay_id"] + feature_cols].rename(
            columns={c: f"{prefix}{c}" for c in feature_cols}
        )
        prefixed.append(renamed)

    # Inner join por stay_id → só pacientes com todos os 4 snapshots (LOS ≥ 96h)
    trajectory = prefixed[0]
    for p in prefixed[1:]:
        trajectory = trajectory.merge(p, on="stay_id", how="inner")

    # Reintegra colunas de meta do D3
    d3_meta = snaps["D3"]["raw"][["stay_id"] + [c for c in meta_cols if c != "stay_id"]]
    trajectory = trajectory.merge(d3_meta, on="stay_id", how="left")

    n_pos = trajectory["readmitted"].sum()
    print(f"\nDataset trajetória: {len(trajectory):,} stays | "
          f"{n_pos:,} positivos ({n_pos/len(trajectory)*100:.1f}%) | "
          f"{trajectory.shape[1]} colunas totais")
    return trajectory


def build_d3_baseline(trajectory_stay_ids):
    """
    D3 retreinado na mesma coorte do modelo de trajetória (LOS ≥ 96h).
    Comparação justa: mesmo conjunto de pacientes.
    """
    path = os.path.join(SNAPSHOTS_DIR, "snapshot_d3.csv")
    d0   = pd.read_csv(os.path.join(SNAPSHOTS_DIR, "snapshot_d0.csv"))
    d1   = pd.read_csv(os.path.join(SNAPSHOTS_DIR, "snapshot_d1.csv"))
    d2   = pd.read_csv(os.path.join(SNAPSHOTS_DIR, "snapshot_d2.csv"))
    d3   = pd.read_csv(path)

    d1 = compute_deltas(d1, d0)
    d2 = compute_deltas(d2, d1)
    d3 = compute_deltas(d3, d2)

    return d3[d3["stay_id"].isin(trajectory_stay_ids)].copy()

# ── Treino e avaliação ────────────────────────────────────────────────────────

def train_and_evaluate(df, label, params_xgb=None, params_lgb=None, is_trajectory=False):
    """
    Treina 4 modelos + Ensemble. Usa params do v7 se disponíveis,
    senão usa defaults conservadores com alta regularização.
    is_trajectory=True: skip engineer_features (já aplicado antes do prefixo)
                        e usa regularização extra para 405 features.
    """
    print(f"\n{'='*65}")
    print(f"  Treinando: {label}")
    print(f"{'='*65}")

    train_df, test_df = patient_level_split(df)
    y_train = train_df["readmitted"].values
    y_test  = test_df["readmitted"].values
    pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"  Treino: {len(train_df):,} | Teste: {len(test_df):,} | pos_weight: {pw:.1f}")

    scaler = StandardScaler()
    X_train, X_train_s, enc, scaler = prepare_X(
        train_df, scaler=scaler, fit=True, skip_engineer=is_trajectory)
    X_test_raw, X_test_s, _, _ = prepare_X(
        test_df, encoders=enc, scaler=scaler, fit=False, skip_engineer=is_trajectory)
    for col in X_train.columns:
        if col not in X_test_raw.columns:
            X_test_raw[col] = 0
            X_test_s[col]   = 0
    X_test_raw = X_test_raw[X_train.columns]
    X_test_s   = X_test_s[X_train.columns]
    print(f"  Features: {X_train.shape[1]}")

    if is_trajectory:
        # 405 features com 14k amostras: regularização mais forte que D3 individual
        xgb_p = {
            "n_estimators": 500, "max_depth": 3, "learning_rate": 0.008,
            "subsample": 0.5, "colsample_bytree": 0.3,
            "min_child_weight": 50, "gamma": 1.0,
            "reg_alpha": 5.0, "reg_lambda": 5.0,
        }
        lgb_p = {
            "n_estimators": 500, "num_leaves": 8, "max_depth": 3,
            "learning_rate": 0.008, "subsample": 0.5, "colsample_bytree": 0.3,
            "min_child_samples": 120, "reg_alpha": 5.0, "reg_lambda": 5.0,
        }
        lr_C = 0.01   # LogReg mais regularizado
    else:
        xgb_p = params_xgb or {
            "n_estimators": 400, "max_depth": 3, "learning_rate": 0.01,
            "subsample": 0.8, "colsample_bytree": 0.6,
            "min_child_weight": 30, "gamma": 0.5,
            "reg_alpha": 3.0, "reg_lambda": 2.0,
        }
        lgb_p = params_lgb or {
            "n_estimators": 400, "num_leaves": 16, "max_depth": 3,
            "learning_rate": 0.01, "subsample": 0.6, "colsample_bytree": 0.6,
            "min_child_samples": 80, "reg_alpha": 2.0, "reg_lambda": 4.0,
        }
        lr_C = 0.05

    models = {
        "LogReg": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, C=lr_C,
        ).fit(X_train_s, y_train),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=5 if is_trajectory else 6,
            min_samples_leaf=20 if is_trajectory else 15,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ).fit(X_train, y_train),
        "XGBoost": xgb.XGBClassifier(
            **xgb_p, scale_pos_weight=pw,
            random_state=42, n_jobs=-1, eval_metric="auc", verbosity=0,
        ).fit(X_train, y_train),
        "LightGBM": lgb.LGBMClassifier(
            **lgb_p, scale_pos_weight=pw,
            random_state=42, n_jobs=-1, verbose=-1,
        ).fit(X_train, y_train),
    }

    results = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "LogReg" else X_train
        X_te = X_test_s  if name == "LogReg" else X_test_raw
        results[name] = compute_metrics(model, X_te, y_test, X_tr, y_train)

    results["Ensemble"] = ensemble_metrics(
        results, y_test, y_train, X_train,
        {"XGBoost": models["XGBoost"], "LightGBM": models["LightGBM"]},
    )

    print(f"\n  {'Modelo':<14} {'AUC':>6} {'AUPRC':>6} {'Brier':>6} {'Recall':>7} {'Overfit':>8}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        ov = f"{r.get('overfitting', 0):+.3f}"
        print(f"  {name:<14} {r['auc']:>6.3f} {r['auprc']:>6.3f} {r['brier']:>6.3f} "
              f"{r['recall']:>7.3f} {ov:>8}")

    # Plots
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} ({res['auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec, prec, lw=2, label=f"{name} ({res['auprc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    slug = label.replace("→", "_").replace(" ", "_")
    axes[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC — {label}")
    axes[0].legend(fontsize=9)
    axes[1].axhline(y_test.mean(), color="gray", ls="--", alpha=0.5)
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"PR — {label}")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"roc_{slug}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({
        "models": models, "encoders": enc, "scaler": scaler,
        "features": X_train.columns.tolist(),
        "metrics": {n: {k: v for k, v in r.items() if k not in ("y_prob", "y_pred")}
                    for n, r in results.items()},
    }, os.path.join(RESULTS_DIR, f"models_{slug.lower()}.joblib"))

    rows = []
    for name, r in results.items():
        rows.append({
            "snapshot": label, "model": name,
            "train_n": len(train_df), "test_n": len(test_df),
            "auc":         round(r["auc"], 4),
            "auprc":       round(r["auprc"], 4),
            "brier":       round(r["brier"], 4),
            "recall":      round(r["recall"], 4),
            "precision":   round(r["precision"], 4),
            "f1":          round(r["f1"], 4),
            "overfitting": round(r.get("overfitting", 0), 4),
        })
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("MODELO DE TRAJETÓRIA D0→D3")
    print("Linha adicional na tabela comparativa de snapshots")
    print("=" * 65)

    # Tenta carregar params do v7 (D3) para usar nos modelos de trajetória
    params_xgb, params_lgb = None, None
    v7_params_path = os.path.join(RESULTS_V7, "best_params_per_snapshot.json")
    if os.path.exists(v7_params_path):
        import json
        with open(v7_params_path) as f:
            v7_params = json.load(f)
        if "D3" in v7_params:
            params_xgb = {k: v for k, v in v7_params["D3"]["xgb"].items()
                          if k not in ("cv_auc", "scale_pos_weight")}
            params_lgb = {k: v for k, v in v7_params["D3"]["lgb"].items()
                          if k not in ("cv_auc", "scale_pos_weight")}
            print("Usando params do v7 (D3) como ponto de partida.")
    else:
        print("Params do v7 não encontrados — usando defaults conservadores.")

    # Dataset de trajetória
    trajectory_df = build_trajectory_dataset()

    # D3 na mesma coorte (comparação justa)
    d3_cohort = build_d3_baseline(set(trajectory_df["stay_id"]))
    print(f"D3 (mesma coorte): {len(d3_cohort):,} stays")

    # Treina e avalia
    all_rows = []
    all_rows.extend(train_and_evaluate(
        d3_cohort, "D3 (coorte LOS≥96h)", params_xgb, params_lgb, is_trajectory=False))
    all_rows.extend(train_and_evaluate(
        trajectory_df, "D0→D3", params_xgb, params_lgb, is_trajectory=True))

    summary = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "trajectory_summary.csv")
    summary.to_csv(out, index=False)

    # Tabela comparativa final
    print(f"\n{'='*65}")
    print("COMPARATIVO: D3 (coorte LOS≥96h)  vs  D0→D3 (trajetória)")
    print("=" * 65)
    pivot = summary.pivot_table(
        index="model", columns="snapshot", values="auc", aggfunc="first"
    ).round(3)
    print(pivot.to_string())

    # Delta trajetória vs D3
    d3_col  = "D3 (coorte LOS≥96h)"
    traj_col = "D0→D3"
    if d3_col in pivot.columns and traj_col in pivot.columns:
        print("\nDelta D0→D3 − D3 (positivo = trajetória melhora):")
        for model in pivot.index:
            delta = pivot.loc[model, traj_col] - pivot.loc[model, d3_col]
            print(f"  {model:<14}: {delta:+.3f}")

    # Mescla com resumo do v7 se existir
    v7_summary_path = os.path.join(RESULTS_V7, "training_summary.csv")
    if os.path.exists(v7_summary_path):
        v7_summary = pd.read_csv(v7_summary_path)
        full = pd.concat([v7_summary, summary], ignore_index=True)
        full_pivot = full.pivot_table(
            index="model", columns="snapshot", values="auc", aggfunc="first"
        ).round(3)
        col_order = [c for c in ["D0", "D1", "D2", "D3", "D3 (coorte LOS≥96h)", "D0→D3"]
                     if c in full_pivot.columns]
        full_pivot = full_pivot[col_order]
        print(f"\n{'='*65}")
        print("TABELA COMPLETA — Todos os snapshots + trajetória")
        print("=" * 65)
        print(full_pivot.to_string())
        full_out = os.path.join(RESULTS_DIR, "full_comparison_table.csv")
        full_pivot.to_csv(full_out)
        print(f"\nTabela completa salva: {full_out}")

    print(f"\nResumo trajetória salvo: {out}")


if __name__ == "__main__":
    main()
