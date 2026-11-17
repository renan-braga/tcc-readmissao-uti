"""
Explicabilidade SHAP — modelo final v7 (per-snapshot tuning)
MIMIC-IV v3.1

Carrega o modelo salvo por 05_tuning_v7_per_snapshot.py e reproduz exatamente o
mesmo conjunto de teste (patient-level split, RANDOM_STATE=42) para calcular
TreeSHAP. A reprodução é validada comparando o AUC recalculado com o AUC salvo
no artefato — se divergir, o script aborta.

Saídas (scripts/resultados/baseline_v7_per_snapshot/shap/):
  - shap_beeswarm_<SNAP>_<MODELO>.png      importância global + direção
  - shap_bar_<SNAP>_<MODELO>.png           |SHAP| médio (top 20)
  - shap_waterfall_..._readmitido.png      explicação individual (verdadeiro positivo)
  - shap_waterfall_..._nao_readmitido.png  explicação individual (verdadeiro negativo)
  - shap_dependence_top4_...png            valor da feature × impacto no risco
  - shap_importance_<SNAP>_<MODELO>.csv    ranking completo com direção do efeito

Uso:
    python 07_shap_v7.py                      # padrão: D0 / XGBoost (modelo final)
    python 07_shap_v7.py --snapshot D1 --model LightGBM

Autor: Renan de Almeida Simões Braga
"""

import os
import re
import sys
import argparse
import importlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

BASE_DIR      = "/Users/renanbraga/tcc_ccai"
SCRIPTS_DIR   = os.path.join(BASE_DIR, "scripts", "features_brutas")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "scripts")
RESULTS_DIR   = None  # herdado do script de treino, logo abaixo
SHAP_DIR      = None
APP_DIR       = os.path.join(BASE_DIR, "app")

SNAPSHOT_ORDER = ["D0", "D1", "D2", "D3"]
TREE_MODELS    = ["XGBoost", "LightGBM", "RandomForest"]

# Reaproveita o pipeline do v7 — garante que o X de teste é idêntico ao do treino
sys.path.insert(0, SCRIPTS_DIR)
v7 = importlib.import_module("05_tuning_v7_per_snapshot")
RESULTS_DIR = v7.RESULTS_DIR
SHAP_DIR    = os.path.join(RESULTS_DIR, "shap")

# Rótulos clínicos em português (dicionário já usado pela API)
sys.path.insert(0, APP_DIR)
import translations as T


# ── Reconstrução do conjunto de teste ────────────────────────────────────────

def build_snapshot_df(snapshot):
    """Reaplica a cadeia de deltas D0→snapshot exatamente como no main() do v7."""
    prev_df = None
    df = None
    for name in SNAPSHOT_ORDER[: SNAPSHOT_ORDER.index(snapshot) + 1]:
        df = pd.read_csv(os.path.join(SNAPSHOTS_DIR, f"snapshot_{name.lower()}.csv"))
        if prev_df is not None:
            df = v7.compute_deltas(df, prev_df)
        prev_df = df.copy()
    return df


def rebuild_test_set(snapshot, artifact):
    df = build_snapshot_df(snapshot)
    _, test_df = v7.patient_level_split(df)

    X_test, _ = v7.prepare_X(test_df, encoders=artifact["encoders"])
    feats = artifact["features"]
    for col in feats:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feats]
    y_test = test_df["readmitted"].values
    return X_test, y_test, test_df



# ── Localização das figuras ──────────────────────────────────────────────────
# A biblioteca shap injeta rótulos em inglês e a fonte padrão do matplotlib não
# possui os subscritos unicode (FiO₂ vira tofu) — ambos corrigidos aqui.

PT_STRINGS = {"Feature value": "Valor da variável", "High": "Alto", "Low": "Baixo"}
SUBSCRITOS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def rotulo_figura(nome):
    return T.translate(nome).translate(SUBSCRITOS)


def localizar(fig):
    for ax in fig.axes:
        if ax.get_ylabel() in PT_STRINGS:
            ax.set_ylabel(PT_STRINGS[ax.get_ylabel()])
        novos, mudou = [], False
        for t in ax.get_yticklabels():
            txt = t.get_text()
            if txt in PT_STRINGS:
                txt, mudou = PT_STRINGS[txt], True
            m = re.match(r"^Sum of (\d+) other features$", txt)
            if m:
                txt, mudou = f"Soma de {m.group(1)} outras variáveis", True
            m = re.match(r"^(\d+) other features$", txt)
            if m:
                txt, mudou = f"{m.group(1)} outras variáveis", True
            novos.append(txt)
        if mudou:
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(novos)


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_test)
    # Classificadores binários de alguns backends devolvem (n, f, 2) — fica a classe 1
    if sv.values.ndim == 3:
        sv = sv[..., 1]
    return sv


def save_beeswarm(sv, snapshot, model_name, auc):
    plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(sv, max_display=20, show=False)
    plt.title(
        f"SHAP — Importância Global das Variáveis\n"
        f"{snapshot} | {model_name} (v7) | AUC = {auc:.3f}",
        fontsize=12,
    )
    plt.xlabel("Impacto no risco predito de readmissão (valor SHAP)")
    localizar(plt.gcf())
    plt.tight_layout()
    path = os.path.join(SHAP_DIR, f"shap_beeswarm_{snapshot}_{model_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_bar(sv, snapshot, model_name):
    plt.subplots(figsize=(9, 8))
    shap.plots.bar(sv, max_display=20, show=False)
    plt.title(f"SHAP — |Impacto| médio\n{snapshot} | {model_name} (v7)", fontsize=12)
    localizar(plt.gcf())
    plt.tight_layout()
    path = os.path.join(SHAP_DIR, f"shap_bar_{snapshot}_{model_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_waterfalls(sv, y_test, y_prob, snapshot, model_name):
    """Um verdadeiro positivo (maior risco entre readmitidos) e um verdadeiro
    negativo (menor risco entre não-readmitidos)."""
    paths = []
    pos = np.where(y_test == 1)[0]
    neg = np.where(y_test == 0)[0]
    escolhas = [
        (pos[np.argmax(y_prob[pos])], "readmitido", "Paciente readmitido em 72h"),
        (neg[np.argmin(y_prob[neg])], "nao_readmitido", "Paciente não readmitido"),
    ]
    for idx, slug, titulo in escolhas:
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(sv[idx], max_display=15, show=False)
        plt.title(
            f"SHAP — Explicação Individual\n{titulo} | {snapshot} | "
            f"{model_name} | risco predito = {y_prob[idx]:.3f}",
            fontsize=11,
        )
        localizar(plt.gcf())
        plt.tight_layout()
        path = os.path.join(
            SHAP_DIR, f"shap_waterfall_{snapshot}_{model_name}_{slug}.png"
        )
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def save_dependence(sv, X_test, ranking, snapshot, model_name, cat_cols=(), top_n=4):
    """Valor da variável × contribuição para o risco — mostra a direção do efeito."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    num = ranking[~ranking["feature"].isin(list(cat_cols))]
    for ax, (_, row) in zip(axes.ravel(), num.head(top_n).iterrows()):
        j = X_test.columns.get_loc(row["feature"])
        x = X_test.iloc[:, j].values
        y = sv.values[:, j]
        ax.scatter(x, y, s=6, alpha=0.25, color="#1f5fa8", edgecolors="none")
        ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.7)
        ax.set_xlabel(row["rotulo_pt"].translate(SUBSCRITOS), fontsize=10)
        ax.set_ylabel("Valor SHAP", fontsize=10)
        ax.set_title(
            f"{row['rotulo_pt'].translate(SUBSCRITOS)}  "
            f"(ρ = {row['spearman_valor_vs_shap']:+.2f})",
            fontsize=10,
        )
    fig.suptitle(
        f"SHAP — Direção do efeito das {top_n} variáveis numéricas mais importantes\n"
        f"{snapshot} | {model_name} (v7)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(SHAP_DIR, f"shap_dependence_top4_{snapshot}_{model_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def build_ranking(sv, X_test, feats, cat_cols=()):
    """Ranking por |SHAP| médio + direção do efeito (Spearman valor × SHAP).

    Variáveis nominais (label-encoded) ficam sem direção: a ordem dos códigos é
    arbitrária (alfabética), então uma correlação monotônica sobre o código não
    tem significado clínico. Para essas, ver shap_categoricas_*.csv.
    """
    vals = sv.values
    rows = []
    for j, f in enumerate(feats):
        x = X_test.iloc[:, j].values
        s = vals[:, j]
        rho = np.nan
        if f not in cat_cols and np.std(x) > 0 and np.std(s) > 0:
            rho = float(spearmanr(x, s).statistic)
        rows.append({
            "feature": f,
            "rotulo_pt": T.translate(f),
            "shap_abs_medio": float(np.abs(s).mean()),
            "shap_medio": float(s.mean()),
            "spearman_valor_vs_shap": rho,
        })
    rk = pd.DataFrame(rows).sort_values("shap_abs_medio", ascending=False)
    rk.insert(0, "rank", range(1, len(rk) + 1))
    rk["direcao"] = np.where(
        rk["feature"].isin(list(cat_cols)), "nominal (ver shap_categoricas_*.csv)",
        np.where(
            rk["spearman_valor_vs_shap"] > 0.05, "maior valor -> maior risco",
            np.where(rk["spearman_valor_vs_shap"] < -0.05,
                     "maior valor -> menor risco", "nao monotonico"),
        ),
    )
    return rk.reset_index(drop=True)


def categorias_shap(sv, X_test, y_test, feats, encoders, col, min_n=0):
    """SHAP médio e readmissão observada por categoria de uma variável nominal."""
    le = encoders[col]
    j = feats.index(col)
    linhas = []
    for code, nome in enumerate(le.classes_):
        m = X_test[col].values == code
        if m.sum() <= min_n:
            continue
        linhas.append({
            "variavel": col, "codigo": int(code), "categoria": str(nome),
            "n": int(m.sum()),
            "shap_medio": float(sv.values[m, j].mean()),
            "readm_observada": float(y_test[m].mean()),
        })
    return pd.DataFrame(linhas).sort_values("shap_medio", ascending=False)


def sigla(nome):
    m = re.search(r"\(([^)]+)\)\s*$", nome)
    return m.group(1) if m else nome


def save_unidades(tab, snapshot, model_name, min_n=30):
    """Barras divergentes: SHAP médio por unidade de UTI de admissão.

    Codificação dupla do sinal (posição em relação ao zero + cor), rótulo direto
    com a readmissão observada — sem segundo eixo.
    """
    d = tab[tab["n"] >= min_n].sort_values("shap_medio")
    if d.empty:
        return None
    VERMELHO, AZUL = "#e34948", "#2a78d6"   # par divergente (quente/frio)
    TINTA, TINTA2  = "#0b0b0b", "#52514e"

    fig, ax = plt.subplots(figsize=(9, 0.52 * len(d) + 2.2))
    cores = [VERMELHO if v > 0 else AZUL for v in d["shap_medio"]]
    y = np.arange(len(d))
    ax.barh(y, d["shap_medio"], height=0.62, color=cores, linewidth=0)
    ax.axvline(0, color="#8c8c88", lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{sigla(c)}  (n={n:,})".replace(",", ".")
                        for c, n in zip(d["categoria"], d["n"])], fontsize=10)
    for yi, (v, r) in enumerate(zip(d["shap_medio"], d["readm_observada"])):
        desloc = 0.006 if v > 0 else -0.006
        ax.text(v + desloc, yi, f"{r*100:.1f}%".replace(".", ","),
                va="center", ha="left" if v > 0 else "right",
                fontsize=9, color=TINTA2)

    lim = float(np.abs(d["shap_medio"]).max()) * 1.45
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("SHAP médio — impacto no risco predito de readmissão",
                  fontsize=10, color=TINTA)
    ax.set_title(
        "Risco aprendido por unidade de admissão na UTI\n"
        f"{snapshot} | {model_name} (v7) | rótulo = readmissão observada em 72h",
        fontsize=12, color=TINTA, loc="left",
    )
    ax.grid(axis="x", color="#e6e5e2", lw=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c4")
    ax.tick_params(colors=TINTA2)
    fig.text(0.01, 0.01,
             "vermelho = aumenta o risco predito | azul = reduz  ·  "
             f"unidades com n ≥ {min_n} no conjunto de teste",
             fontsize=8.5, color=TINTA2)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(SHAP_DIR, f"shap_unidades_{snapshot}_{model_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SHAP do modelo final v7")
    ap.add_argument("--snapshot", default="D0", choices=SNAPSHOT_ORDER)
    ap.add_argument("--model", default="XGBoost", choices=TREE_MODELS)
    args = ap.parse_args()

    os.makedirs(SHAP_DIR, exist_ok=True)
    snapshot, model_name = args.snapshot, args.model

    print("=" * 68)
    print(f"SHAP — {snapshot} | {model_name} | baseline v7 (per-snapshot tuning)")
    print("=" * 68)

    art_path = os.path.join(RESULTS_DIR, f"models_{snapshot.lower()}.joblib")
    artifact = joblib.load(art_path)
    model = artifact["models"][model_name]
    feats = artifact["features"]
    auc_salvo = artifact["metrics"][model_name]["auc"]
    print(f"  Artefato: {os.path.basename(art_path)} | {len(feats)} features")

    print("  Reconstruindo conjunto de teste (patient-level split, seed 42)...")
    X_test, y_test, test_df = rebuild_test_set(snapshot, artifact)
    print(f"    Teste: {len(X_test):,} estadias | "
          f"{int(y_test.sum()):,} readmissões ({y_test.mean():.3%})")

    # Validação: o teste reconstruído tem de reproduzir o AUC salvo
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_novo = roc_auc_score(y_test, y_prob)
    print(f"    AUC salvo = {auc_salvo:.6f} | AUC recalculado = {auc_novo:.6f}")
    if abs(auc_novo - auc_salvo) > 1e-6:
        print("\n  ERRO: o conjunto de teste não foi reproduzido — SHAP abortado.")
        print("  Verifique se os snapshots CSV mudaram desde o treino do v7.")
        sys.exit(1)
    print("    Split reproduzido exatamente.\n")

    print("  Calculando TreeSHAP...")
    sv = compute_shap(model, X_test)
    cat_cols = [c for c in artifact["encoders"] if c in feats]
    ranking = build_ranking(sv, X_test, feats, cat_cols)

    csv_path = os.path.join(SHAP_DIR, f"shap_importance_{snapshot}_{model_name}.csv")
    ranking.to_csv(csv_path, index=False)

    # Variáveis nominais: efeito por categoria (forma interpretável correta)
    tabelas = [categorias_shap(sv, X_test, y_test, feats, artifact["encoders"], c)
               for c in cat_cols]
    tab_cat = pd.concat(tabelas, ignore_index=True)
    cat_csv = os.path.join(SHAP_DIR, f"shap_categoricas_{snapshot}_{model_name}.csv")
    tab_cat.to_csv(cat_csv, index=False)
    unidades = tab_cat[tab_cat["variavel"] == "first_careunit"]

    # Rótulos em português apenas para as figuras
    sv.feature_names = [rotulo_figura(f) for f in feats]

    print("  Gerando figuras...")
    paths = [
        save_beeswarm(sv, snapshot, model_name, auc_novo),
        save_bar(sv, snapshot, model_name),
        *save_waterfalls(sv, y_test, y_prob, snapshot, model_name),
        save_dependence(sv, X_test, ranking, snapshot, model_name, cat_cols),
    ]
    p_un = save_unidades(unidades, snapshot, model_name)
    if p_un:
        paths.append(p_un)

    print("\n" + "=" * 68)
    print(f"TOP 15 — |SHAP| médio ({snapshot} | {model_name})")
    print("=" * 68)
    print(ranking.head(15)[
        ["rank", "rotulo_pt", "shap_abs_medio", "spearman_valor_vs_shap", "direcao"]
    ].to_string(index=False))

    d_un = unidades[unidades["n"] >= 30].sort_values("shap_medio", ascending=False)
    if not d_un.empty:
        rho_un = float(np.corrcoef(d_un["shap_medio"], d_un["readm_observada"])[0, 1])
        print("\n" + "=" * 68)
        print(f"Unidade de admissão — SHAP × readmissão observada (r = {rho_un:.2f})")
        print("=" * 68)
        print(d_un[["categoria", "n", "shap_medio", "readm_observada"]]
              .round(4).to_string(index=False))

    print("\nArquivos salvos:")
    for p in paths + [csv_path, cat_csv]:
        print(f"  {os.path.relpath(p, BASE_DIR)}")


if __name__ == "__main__":
    main()
