"""
Treinamento e Avaliação — Readmissão UTI 72h — MIMIC-IV v3.1

Modelos: Logistic Regression, Random Forest, XGBoost, LightGBM
Explicabilidade: SHAP (TreeSHAP para modelos de árvore; beeswarm global + waterfall individual)
Validação: patient-level split 80/20 — todas as stays de um paciente vão para o mesmo
           conjunto (treino ou teste). Estratificado por "paciente teve ao menos 1 readmissão".
           Necessário porque E10 foi removido: múltiplas stays por paciente existem no dataset.
           Split por linha causaria leakage (mesmo paciente em treino e teste).
Métricas: AUC-ROC, AUPRC, Brier Score, Recall, Precision, F1

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

# ── Configuração ──────────────────────────────────────────────────────────────

SCRIPTS_DIR  = "/Users/renanbraga/tcc_ccai/scripts"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
TARGET_RECALL = 0.60
TARGET_AUC    = 0.75

# Colunas administrativas/identificadoras — nunca entram como features
ID_COLS = [
    "stay_id", "subject_id", "hadm_id",
    "intime", "outtime", "admittime", "dischtime",
    "deathtime", "next_icu_intime",
    "readmitted",
    "hours_until_readmission",  # leakage direto
    "discharge_location",       # leakage indireto (já conhecer a alta)
    "icu_los_hours",            # leakage: é o LOS TOTAL, desconhecido no corte (set/2026)
    "last_careunit",            # duplicata exata de first_careunit (set/2026)
    "anchor_year", "anchor_year_group", "stay_rank",
    "admittime", "dischtime",
]

# Componentes SOFA excluídos: são derivados das variáveis brutas que JÁ entram
# no modelo. Incluir ambos fragmentaria o SHAP sem adicionar informação.
# O sofa_total entra como feature âncora de referência clínica.
SOFA_COMPONENTS = [
    "sofa_coagulation", "sofa_liver", "sofa_renal",
    "sofa_neurologic", "sofa_cardiovascular", "sofa_respiratory",
]

# Colunas usadas para calcular deltas entre snapshots consecutivos (Dn − D(n-1)).
# delta_* NÃO são excluídas pelo SOFA_COMPONENTS ou ID_COLS pois têm prefixo "delta_".
#
# Grupo A — Point-in-time (janela rolante 24h de mimiciv_derived.sofa):
#   delta = Dn - D(n-1) é sinal REAL de trajetória (melhora < 0, piora > 0)
#
# Grupo B — Cumulativas (max/min/mean sobre 0→Nh):
#   delta ≥ 0 para max, ≤ 0 para min — representa "novo extremo na janela adicional"
#   sinal mais fraco que Grupo A, mas ainda informativo
DELTA_COLS = [
    # Grupo A: point-in-time (trajetória real)
    "sofa_pt",
    "sofa_pt_respiration", "sofa_pt_coagulation", "sofa_pt_liver",
    "sofa_pt_cardiovascular", "sofa_pt_neurologic", "sofa_pt_renal",
    "gcs_pt", "creatinine_pt", "platelet_pt", "bilirubin_pt",
    "meanbp_pt", "urine_output_pt", "pao2fio2_pt",
    # Grupo B: cumulativas
    "sofa_total",
    "creatinine_max", "creatinine_mean",
    "bilirubin_max",
    "map_mean",
    "heart_rate_mean",
    "gcs_total_min",
    "urine_output_ml",
    "po2_min",
    "bicarbonate_mean",
    "wbc_max",
    "norepinephrine_max",
    "any_vasopressor_max",
    "spo2_min",
    "respiratory_rate_mean",
    "systolic_bp_mean",
    "potassium_max",
]

# ── Features de trajetória (delta entre snapshots) ───────────────────────────

def compute_deltas(curr_df, prev_df):
    """Adiciona colunas delta_* ao curr_df comparando com prev_df (snapshot anterior).

    Stays presentes em curr mas ausentes em prev recebem NaN → preenchidos com 0
    em prepare_X. Isso ocorre quando uma internação atinge o limiar de Dn pela
    primeira vez sem ter tido leitura em D(n-1).
    """
    cols = [c for c in DELTA_COLS if c in curr_df.columns and c in prev_df.columns]
    prev_subset = prev_df[["stay_id"] + cols].rename(
        columns={c: f"_prev_{c}" for c in cols}
    )
    merged = curr_df.merge(prev_subset, on="stay_id", how="left")
    for c in cols:
        merged[f"delta_{c}"] = merged[c] - merged[f"_prev_{c}"]
    merged = merged.drop(columns=[f"_prev_{c}" for c in cols])
    return merged


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
                df[col] = df[col].apply(
                    lambda x: int(le.transform([x])[0]) if x in known else -1
                )
    return df, encoders


def engineer_features(df):
    """Features clínicas derivadas sem informação futura."""
    df = df.copy()
    if "heart_rate_mean" in df.columns and "systolic_bp_mean" in df.columns:
        df["shock_index"] = df["heart_rate_mean"] / df["systolic_bp_mean"].clip(lower=1)
        df["severe_hypotension"] = (df["systolic_bp_mean"] < 90).astype(int)
        df["tachycardia"]        = (df["heart_rate_mean"] > 100).astype(int)

    if "po2_min" in df.columns and "fio2_max" in df.columns:
        fio2 = df["fio2_max"].where(df["fio2_max"] > 1, df["fio2_max"] * 100)
        df["pf_ratio"]           = df["po2_min"] / fio2.clip(lower=21) * 100
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


def prepare_X(df, encoders=None, scaler=None, fit=True):
    """Retorna X limpo, encoders e scaler (para LogReg)."""
    drop = [c for c in ID_COLS + SOFA_COMPONENTS if c in df.columns]
    X = df.drop(columns=drop)
    X = engineer_features(X)
    X, encoders = encode_categoricals(X, encoders)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if scaler is not None:
        if fit:
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), columns=X.columns, index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                scaler.transform(X), columns=X.columns, index=X.index
            )
        return X, X_scaled, encoders, scaler
    return X, encoders


def patient_level_split(df):
    """Split 80/20 por subject_id — todas as stays de um paciente vão pro mesmo conjunto."""
    # Estratificar por "paciente teve ao menos 1 readmissão" para balancear positivos
    patient_label = df.groupby("subject_id")["readmitted"].max()
    train_patients, test_patients = train_test_split(
        patient_label.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=patient_label.values,
    )
    train = df[df["subject_id"].isin(train_patients)].copy()
    test  = df[df["subject_id"].isin(test_patients)].copy()
    return train, test

# ── Builders de modelos ───────────────────────────────────────────────────────

def build_logreg(X_train_scaled, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        C=0.1,
    )
    model.fit(X_train_scaled, y_train)
    return model


def build_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def build_xgboost(X_train, y_train):
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def build_lightgbm(X_train, y_train):
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model

# ── Métricas e threshold ──────────────────────────────────────────────────────

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

# ── SHAP ─────────────────────────────────────────────────────────────────────

def run_shap(model, X_test, y_test, y_prob, model_name, snapshot_name, out_dir):
    """Gera beeswarm global + waterfall de 2 pacientes (1 readmitido, 1 não)."""
    print(f"    SHAP para {model_name}...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Beeswarm global ---
    plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP — Importância Global\n{snapshot_name} | {model_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"shap_beeswarm_{snapshot_name}_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Waterfall: 1 paciente readmitido (prob alta) + 1 não-readmitido (prob baixa) ---
    positive_idx = np.where(y_test == 1)[0]
    negative_idx = np.where(y_test == 0)[0]

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        return

    patient_pos = positive_idx[np.argmax(y_prob[positive_idx])]
    patient_neg = negative_idx[np.argmin(y_prob[negative_idx])]

    for idx, label in [(patient_pos, "readmitido"), (patient_neg, "nao_readmitido")]:
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
        plt.title(
            f"SHAP — Explicação Individual ({label})\n"
            f"{snapshot_name} | {model_name} | P={y_prob[idx]:.2f}",
            fontsize=11,
        )
        plt.tight_layout()
        path = os.path.join(
            out_dir, f"shap_waterfall_{snapshot_name}_{model_name}_{label}.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"      Salvo: beeswarm + 2 waterfalls")


def plot_logreg_or(model, feature_names, snapshot_name, out_dir, top_n=20):
    """Forest plot de odds ratios para Logistic Regression."""
    coefs = model.coef_[0]
    OR    = np.exp(coefs)
    df_or = pd.DataFrame({"feature": feature_names, "OR": OR, "coef": coefs})
    df_or = df_or.reindex(df_or["OR"].abs().sort_values(ascending=False).index)
    df_or = df_or.head(top_n).sort_values("OR")

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))
    colors = ["#d73027" if v > 1 else "#4575b4" for v in df_or["OR"]]
    ax.barh(df_or["feature"].tolist(), (df_or["OR"] - 1).tolist(), left=1, color=colors, alpha=0.8)
    ax.axvline(1, color="black", lw=1, ls="--")
    ax.set_xlabel("Odds Ratio (por DP)")
    fig.suptitle(f"Odds Ratios — Logistic Regression\n{snapshot_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"logreg_or_{snapshot_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return df_or

# ── Plot comparativo de todos os modelos ─────────────────────────────────────

def plot_roc_comparison(results_dict, y_test, snapshot_name, out_dir):
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC={res['auc']:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec, prec, lw=2, label=f"{name} (AUPRC={res['auprc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC — {snapshot_name}")
    axes[0].legend(fontsize=9)

    axes[1].axhline(y_test.mean(), color="gray", ls="--", alpha=0.5,
                    label=f"Baseline={y_test.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall — {snapshot_name}")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f"roc_comparison_{snapshot_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# ── Pipeline por snapshot ─────────────────────────────────────────────────────

def train_snapshot(snapshot_name, filepath, prev_df=None):
    print(f"\n{'='*60}")
    print(f"  Snapshot {snapshot_name}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath)
    raw_df = df.copy()  # preservado sem deltas para ser usado como prev_df no próximo snapshot

    if prev_df is not None:
        df = compute_deltas(df, prev_df)
        n_delta = sum(1 for c in df.columns if c.startswith("delta_"))
        print(f"  Features delta adicionadas: {n_delta}")

    print(f"  {df.shape[0]:,} linhas | {df['readmitted'].mean():.3f} taxa readmissão")

    train_df, test_df = patient_level_split(df)
    print(f"  Treino: {len(train_df):,} | Teste: {len(test_df):,}")

    y_train = train_df["readmitted"].values
    y_test  = test_df["readmitted"].values

    # Preparar features — versão padrão (tree models) e escalada (LogReg)
    scaler = StandardScaler()
    X_train, X_train_s, enc, scaler = prepare_X(train_df, scaler=scaler, fit=True)
    X_test_raw, X_test_s, _, _      = prepare_X(test_df, encoders=enc, scaler=scaler, fit=False)

    # Alinhar colunas entre treino e teste
    for col in X_train.columns:
        if col not in X_test_raw.columns:
            X_test_raw[col] = 0
            X_test_s[col]   = 0
    X_test_raw = X_test_raw[X_train.columns]
    X_test_s   = X_test_s[X_train.columns]

    print(f"  Features: {X_train.shape[1]}")

    # ── Treinar modelos ───────────────────────────────────────────────────────
    models = {}
    print("\n  Treinando modelos...")

    print("    Logistic Regression...", end=" ")
    models["LogReg"] = build_logreg(X_train_s, y_train)
    print("ok")

    print("    Random Forest...", end=" ")
    models["RandomForest"] = build_rf(X_train, y_train)
    print("ok")

    print("    XGBoost...", end=" ")
    models["XGBoost"] = build_xgboost(X_train, y_train)
    print("ok")

    print("    LightGBM...", end=" ")
    models["LightGBM"] = build_lightgbm(X_train, y_train)
    print("ok")

    # ── Avaliar métricas ──────────────────────────────────────────────────────
    print("\n  Avaliando...")
    results = {}
    for name, model in models.items():
        X_tr = X_train_s if name == "LogReg" else X_train
        X_te = X_test_s  if name == "LogReg" else X_test_raw
        results[name] = compute_metrics(model, X_te, y_test, X_tr, y_train)

    # ── Imprimir tabela ───────────────────────────────────────────────────────
    print(f"\n  {'Modelo':<14} {'AUC':>6} {'AUPRC':>6} {'Brier':>6} {'Recall':>7} {'Prec':>6} {'F1':>6} {'Overfit':>8}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        ov = f"{r.get('overfitting', 0):+.3f}"
        ok = "✅" if r["auc"] >= TARGET_AUC and r["recall"] >= TARGET_RECALL else "❌"
        print(f"  {name:<14} {r['auc']:>6.3f} {r['auprc']:>6.3f} {r['brier']:>6.3f} "
              f"{r['recall']:>7.3f} {r['precision']:>6.3f} {r['f1']:>6.3f} {ov:>8} {ok}")

    # ── SHAP para modelos de árvore ───────────────────────────────────────────
    print("\n  Gerando SHAP (modelos de árvore)...")
    for name in ["XGBoost", "LightGBM"]:
        try:
            run_shap(
                models[name], X_test_raw, y_test,
                results[name]["y_prob"], name, snapshot_name, SCRIPTS_DIR,
            )
        except Exception as e:
            print(f"    SHAP {name} falhou: {e}")

    # ── Odds ratios (LogReg) ──────────────────────────────────────────────────
    print("  Gerando odds ratios (LogReg)...")
    df_or = plot_logreg_or(
        models["LogReg"], X_train.columns.tolist(), snapshot_name, SCRIPTS_DIR
    )
    df_or.to_csv(
        os.path.join(SCRIPTS_DIR, f"logreg_or_{snapshot_name}.csv"), index=False
    )

    # ── ROC/PR comparativo ────────────────────────────────────────────────────
    plot_roc_comparison(results, y_test, snapshot_name, SCRIPTS_DIR)

    # ── Salvar modelos ────────────────────────────────────────────────────────
    joblib.dump({
        "models": models,
        "encoders": enc,
        "scaler": scaler,
        "features": X_train.columns.tolist(),
        "metrics": {name: {k: v for k, v in r.items() if k not in ("y_prob", "y_pred")}
                    for name, r in results.items()},
    }, os.path.join(SCRIPTS_DIR, f"models_{snapshot_name.lower()}.joblib"))

    # Retornar linhas para o summary + raw_df para usar como prev_df no próximo snapshot
    rows = []
    for name, r in results.items():
        rows.append({
            "snapshot":   snapshot_name,
            "model":      name,
            "train_n":    len(train_df),
            "test_n":     len(test_df),
            "auc":        round(r["auc"],       4),
            "auprc":      round(r["auprc"],     4),
            "brier":      round(r["brier"],     4),
            "recall":     round(r["recall"],    4),
            "precision":  round(r["precision"], 4),
            "f1":         round(r["f1"],        4),
            "overfitting":round(r.get("overfitting", 0), 4),
            "threshold":  round(r["threshold"], 3),
            "targets_met": "✅" if r["auc"] >= TARGET_AUC and r["recall"] >= TARGET_RECALL else "❌",
        })
    return rows, raw_df

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TREINAMENTO — READMISSÃO UTI 72h — MIMIC-IV v3.1")
    print(f"Split: patient-level 80/20 (seed={RANDOM_STATE}) — sem leakage entre stays do mesmo paciente")
    print(f"Modelos: LogReg | RandomForest | XGBoost | LightGBM")
    print("=" * 60)

    snapshots = [
        (sid, os.path.join(SCRIPTS_DIR, f"snapshot_{sid.lower()}.csv"))
        for sid in ["D0", "D1", "D2", "D3"]
        if os.path.exists(os.path.join(SCRIPTS_DIR, f"snapshot_{sid.lower()}.csv"))
    ]

    if not snapshots:
        print("Nenhum snapshot encontrado. Rode primeiro o script 02.")
        return

    print(f"Snapshots encontrados: {[s[0] for s in snapshots]}\n")

    all_rows = []
    prev_df = None
    for name, path in snapshots:
        try:
            rows, prev_df = train_snapshot(name, path, prev_df=prev_df)
            if rows:
                all_rows.extend(rows)
        except Exception as e:
            import traceback
            print(f"ERRO no snapshot {name}: {e}")
            traceback.print_exc()
            prev_df = None  # não propagar estado corrompido

    if not all_rows:
        print("Nenhum modelo treinado com sucesso.")
        return

    summary = pd.DataFrame(all_rows)
    out = os.path.join(SCRIPTS_DIR, "training_summary.csv")
    summary.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print("RESUMO FINAL")
    print("=" * 60)
    pivot = summary.pivot_table(
        index="model", columns="snapshot", values="auc", aggfunc="first"
    ).round(3)
    print(pivot.to_string())
    print(f"\nResumo completo salvo: {out}")


if __name__ == "__main__":
    main()
