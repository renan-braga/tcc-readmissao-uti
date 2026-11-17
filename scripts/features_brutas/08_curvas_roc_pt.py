"""
Curvas ROC e Precision-Recall em português — modelos v7
MIMIC-IV v3.1

Regenera a figura de curvas do snapshot escolhido a partir dos modelos salvos,
com rótulos em português, para uso direto no paper. Reaproveita a reconstrução
validada do conjunto de teste de 07_shap_v7.py (o AUC é conferido contra o valor
salvo no artefato antes de plotar).

Uso:
    python 08_curvas_roc_pt.py [--snapshot D0]

Autor: Renan de Almeida Simões Braga
"""

import os
import sys
import argparse
import importlib
import warnings

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")
sns.set_style("whitegrid")

BASE_DIR    = "/Users/renanbraga/tcc_ccai"
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts", "features_brutas")
RESULTS_DIR = None  # definido abaixo, herdado do script de treino

sys.path.insert(0, SCRIPTS_DIR)
shp = importlib.import_module("07_shap_v7")
RESULTS_DIR = shp.RESULTS_DIR

NOMES_PT = {
    "LogReg": "Regressão logística",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "Ensemble": "Ensemble (XGB+LGB)",
}
CORES = ["#52514e", "#1baf7a", "#2a78d6", "#eb6834", "#e34948"]


def num(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser(description="Curvas ROC/PR em português (v7)")
    ap.add_argument("--snapshot", default="D0", choices=["D0", "D1", "D2", "D3"])
    args = ap.parse_args()
    snap = args.snapshot

    art = joblib.load(os.path.join(RESULTS_DIR, f"models_{snap.lower()}.joblib"))
    X_test, y_test, _ = shp.rebuild_test_set(snap, art)
    prevalencia = y_test.mean()

    probs = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cor, (nome, modelo) in zip(CORES, art["models"].items()):
        Xs = X_test
        if nome == "LogReg":                      # LogReg foi treinado em escala padronizada
            Xs = art["scaler"].transform(X_test)
        prob = modelo.predict_proba(Xs)[:, 1]

        auc = roc_auc_score(y_test, prob)
        salvo = art["metrics"][nome]["auc"]
        if abs(auc - salvo) > 1e-6:
            print(f"  ERRO: {nome} não reproduziu o AUC salvo "
                  f"({auc:.6f} vs {salvo:.6f}) — figura abortada.")
            sys.exit(1)

        probs[nome] = prob

        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[0].plot(fpr, tpr, lw=2, color=cor,
                     label=f"{NOMES_PT[nome]} ({num(auc)})")
        prec, rec, _ = precision_recall_curve(y_test, prob)
        axes[1].plot(rec, prec, lw=2, color=cor,
                     label=f"{NOMES_PT[nome]} ({num(art['metrics'][nome]['auprc'])})")

    # Ensemble: média simples das probabilidades XGBoost e LightGBM
    if {"XGBoost", "LightGBM"} <= probs.keys() and "Ensemble" in art["metrics"]:
        prob = (probs["XGBoost"] + probs["LightGBM"]) / 2
        auc = roc_auc_score(y_test, prob)
        salvo = art["metrics"]["Ensemble"]["auc"]
        if abs(auc - salvo) > 1e-6:
            print(f"  ERRO: Ensemble não reproduziu o AUC salvo "
                  f"({auc:.6f} vs {salvo:.6f}) — figura abortada.")
            sys.exit(1)
        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[0].plot(fpr, tpr, lw=2, color=CORES[4],
                     label=f"{NOMES_PT['Ensemble']} ({num(auc)})")
        prec, rec, _ = precision_recall_curve(y_test, prob)
        axes[1].plot(rec, prec, lw=2, color=CORES[4],
                     label=f"{NOMES_PT['Ensemble']} ({num(art['metrics']['Ensemble']['auprc'])})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    axes[0].set_xlabel("Taxa de falsos positivos (1 − especificidade)", fontsize=11)
    axes[0].set_ylabel("Sensibilidade", fontsize=11)
    axes[0].set_title(f"Curva ROC — {snap}", fontsize=12)
    axes[0].legend(fontsize=9.5, loc="lower right", title="AUC-ROC", title_fontsize=9.5)

    axes[1].axhline(prevalencia, color="#8c8c88", ls="--", lw=1,
                    label=f"Prevalência ({num(prevalencia*100, 2)}%)")
    axes[1].set_xlabel("Sensibilidade (recall)", fontsize=11)
    axes[1].set_ylabel("Precisão", fontsize=11)
    axes[1].set_title(f"Curva Precisão–Sensibilidade — {snap}", fontsize=12)
    axes[1].set_ylim(0, max(0.35, prevalencia * 5))
    axes[1].legend(fontsize=9.5, loc="upper right", title="AUPRC", title_fontsize=9.5)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"curvas_{snap}_pt.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  {len(y_test):,} estadias | prevalência {prevalencia:.4%}")
    print(f"  Salvo: {os.path.relpath(out, BASE_DIR)}")


if __name__ == "__main__":
    main()
