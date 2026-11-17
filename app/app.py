"""
Interface Gráfica — Predição de Readmissão em UTI
MIMIC-IV v3.1 | Modelos v7 (Optuna per-snapshot)

Duas abas:
  🔍 Busca por Paciente — stay_id → consulta snapshots locais → predição + SHAP
  📂 Upload CSV         — arquivo com dados do paciente → predição + SHAP
"""

import io
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predição de Readmissão em UTI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Fundo geral */
    .stApp { background-color: #0f1117; }

    /* Cards */
    .risk-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2d3250;
        margin-bottom: 12px;
    }
    .risk-card h3 { color: #a0aec0; font-size: 13px; font-weight: 500;
                    text-transform: uppercase; letter-spacing: 1px; margin: 0 0 6px 0; }
    .risk-card .value { font-size: 36px; font-weight: 700; margin: 0; }

    /* Cores de risco */
    .low    { color: #48bb78; }
    .medium { color: #ed8936; }
    .high   { color: #fc5c65; }

    /* Cabeçalho */
    .header-box {
        background: linear-gradient(135deg, #1a1f35 0%, #2d3250 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        border: 1px solid #3d4470;
    }
    .header-box h1 { color: #e2e8f0; font-size: 26px; margin: 0 0 4px 0; }
    .header-box p  { color: #718096; font-size: 14px; margin: 0; }

    /* Badge de snapshot */
    .snap-badge {
        display: inline-block;
        background: #2d3250;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 12px;
        color: #90cdf4;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Dividers */
    hr { border-color: #2d3250; }

    /* Botões */
    .stButton > button {
        background: #4c6ef5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stButton > button:hover { background: #3b5bdb; }
</style>
""", unsafe_allow_html=True)

# ── Imports internos (com cache do Streamlit) ─────────────────────────────────

import sys
sys.path.insert(0, "/Users/renanbraga/tcc_ccai/app")

import os, joblib
from utils import (
    SNAPSHOT_NAMES, SNAPSHOT_LABELS, SNAPSHOTS_DIR,
    compute_deltas, prepare_X_for_inference, get_template_csv,
    engineer_features, ID_COLS, SOFA_COMPONENTS,
)
from translations import translate, describe

MODELS_DIR = "/Users/renanbraga/tcc_ccai/scripts/resultados/baseline_v7_per_snapshot"


@st.cache_resource(show_spinner="Carregando modelos...")
def load_all_bundles():
    bundles = {}
    for name in SNAPSHOT_NAMES:
        path = os.path.join(MODELS_DIR, f"models_{name.lower()}.joblib")
        bundles[name] = joblib.load(path)
    return bundles


@st.cache_data(show_spinner="Carregando snapshots...")
def load_all_snapshots():
    snaps, prev = {}, None
    for name in SNAPSHOT_NAMES:
        path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{name.lower()}.csv")
        df = pd.read_csv(path)
        if prev is not None:
            df = compute_deltas(df, prev)
        snaps[name] = df
        prev = pd.read_csv(os.path.join(SNAPSHOTS_DIR, f"snapshot_{name.lower()}.csv"))
    return snaps

# ── Helpers de UI ─────────────────────────────────────────────────────────────

def risk_color(prob):
    if prob < 0.30:  return "low",    "🟢", "#48bb78"
    if prob < 0.55:  return "medium", "🟡", "#ed8936"
    return                  "high",   "🔴", "#fc5c65"


def gauge_chart(prob, title="Probabilidade de Readmissão"):
    cls, _, color = risk_color(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 38, "color": color}},
        title={"text": title, "font": {"size": 14, "color": "#a0aec0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568",
                     "tickfont": {"color": "#718096", "size": 11}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1e2130",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#1a2a1a"},
                {"range": [30, 55], "color": "#2a2010"},
                {"range": [55, 100],"color": "#2a1015"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": prob * 100},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
        margin={"t": 60, "b": 20, "l": 30, "r": 30},
        height=240,
        font={"color": "#e2e8f0"},
    )
    return fig


def trajectory_chart(predictions):
    snaps  = list(predictions.keys())
    probs  = [predictions[s]["prob"] * 100 for s in snaps]
    labels = [f"D{i}" for i in range(len(snaps))]

    colors = [risk_color(p / 100)[2] for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=probs,
        mode="lines+markers",
        line={"color": "#4c6ef5", "width": 2.5},
        marker={"size": 12, "color": colors, "line": {"color": "#1e2130", "width": 2}},
        hovertemplate="<b>%{x}</b><br>Risco: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#48bb78", opacity=0.05, line_width=0)
    fig.add_hrect(y0=30, y1=55,  fillcolor="#ed8936", opacity=0.05, line_width=0)
    fig.add_hrect(y0=55, y1=100, fillcolor="#fc5c65", opacity=0.05, line_width=0)
    fig.update_layout(
        title={"text": "Trajetória de Risco por Snapshot", "font": {"size": 14, "color": "#a0aec0"}},
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
        xaxis={"gridcolor": "#2d3250", "color": "#718096"},
        yaxis={"gridcolor": "#2d3250", "color": "#718096",
               "range": [0, 100], "ticksuffix": "%"},
        margin={"t": 50, "b": 30, "l": 40, "r": 20},
        height=260,
        font={"color": "#e2e8f0"},
    )
    return fig


def shap_waterfall(pred, snap_name, top_n=15):
    if pred["shap_values"] is None:
        st.warning("SHAP não disponível para este modelo.")
        return

    sv   = pred["shap_values"]
    fv   = pred["feature_values"]
    fn   = pred["feature_names"]
    base = pred["base_value"]

    # Top N features por |shap|
    idx  = np.argsort(np.abs(sv))[::-1][:top_n]
    sv_t = sv[idx]
    fv_t = fv[idx]
    fn_t = [fn[i] for i in idx]

    # Ordena por valor para waterfall
    order = np.argsort(sv_t)
    sv_t  = sv_t[order]
    fv_t  = fv_t[order]
    fn_t  = [fn_t[i] for i in order]

    # Traduz nomes para português
    labels      = [f"{translate(n)} = {v:.2f}" for n, v in zip(fn_t, fv_t)]
    bar_clrs    = ["#fc5c65" if v > 0 else "#48bb78" for v in sv_t]
    hover_texts = []
    for n, v, s in zip(fn_t, fv_t, sv_t):
        label_pt = translate(n)
        descr    = describe(n)
        direction = "▲ aumenta o risco" if s > 0 else "▼ reduz o risco"
        tooltip = (
            f"<b>{label_pt}</b><br>"
            f"Valor: <b>{v:.3f}</b><br>"
            f"Contribuição SHAP: <b>{s:+.4f}</b> ({direction})<br>"
        )
        if descr:
            # Quebra descrições longas em ~60 chars por linha
            words, line, lines = descr.split(), "", []
            for w in words:
                if len(line) + len(w) + 1 > 60:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)
            tooltip += "<br><i style='color:#a0aec0'>" + "<br>".join(lines) + "</i>"
        hover_texts.append(tooltip)

    fig = go.Figure(go.Bar(
        x=sv_t,
        y=labels,
        orientation="h",
        marker_color=bar_clrs,
        text=["▲" if v > 0 else "▼" for v in sv_t],
        textposition="outside",
        textfont={"size": 10, "color": bar_clrs},
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))
    fig.add_vline(x=0, line_color="#4a5568", line_width=1)
    fig.update_layout(
        title={"text": f"Explicação SHAP — {SNAPSHOT_LABELS.get(snap_name, snap_name)}",
               "font": {"size": 14, "color": "#a0aec0"}},
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
        xaxis={"gridcolor": "#2d3250", "color": "#718096",
               "title": "Contribuição para o risco"},
        yaxis={"color": "#e2e8f0", "tickfont": {"size": 11}},
        margin={"t": 50, "b": 30, "l": 220, "r": 30},
        height=max(350, top_n * 28),
        font={"color": "#e2e8f0"},
    )
    return fig


def predict_and_display(patient_snaps, bundles, model_name="XGBoost"):
    """Roda inferência e exibe resultados."""
    predictions = {}
    for snap_name, row_df in patient_snaps.items():
        bundle = bundles[snap_name]
        model  = bundle["models"][model_name]
        X = prepare_X_for_inference(row_df, bundle)
        prob = float(model.predict_proba(X)[0, 1])
        try:
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(X)
            if isinstance(sv, list): sv = sv[1]
            base = exp.expected_value
            if isinstance(base, (list, np.ndarray)): base = base[1]
        except Exception:
            sv, base = None, None
        predictions[snap_name] = {
            "prob":           prob,
            "shap_values":    sv[0] if sv is not None else None,
            "base_value":     float(base) if base is not None else None,
            "feature_names":  X.columns.tolist(),
            "feature_values": X.values[0],
        }

    # ── Cabeçalho de resultado ────────────────────────────────────────────────
    st.markdown("---")
    best_snap = max(predictions, key=lambda s: predictions[s]["prob"])
    best_prob = predictions[best_snap]["prob"]
    cls, icon, color = risk_color(best_prob)

    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown(f"""
        <div class="risk-card">
            <h3>Risco Máximo Observado</h3>
            <p class="value {cls}">{icon} {best_prob*100:.1f}%
               <span style="font-size:16px;color:#718096;font-weight:400">
                 — {SNAPSHOT_LABELS.get(best_snap, best_snap)}
               </span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Gauge + Trajetória ────────────────────────────────────────────────────
    if len(predictions) == 1:
        snap = list(predictions.keys())[0]
        st.plotly_chart(gauge_chart(predictions[snap]["prob"]),
                        use_container_width=True, key=f"gauge_{snap}")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(gauge_chart(best_prob, "Risco Máximo"),
                            use_container_width=True, key="gauge_max")
        with c2:
            st.plotly_chart(trajectory_chart(predictions),
                            use_container_width=True, key="traj")

    # ── Cards por snapshot ────────────────────────────────────────────────────
    st.markdown("#### Probabilidade por Snapshot")
    cols = st.columns(len(predictions))
    for col, (snap_name, pred) in zip(cols, predictions.items()):
        cls2, icon2, color2 = risk_color(pred["prob"])
        with col:
            st.markdown(f"""
            <div class="risk-card" style="text-align:center">
                <div class="snap-badge">{snap_name}</div>
                <p class="value {cls2}" style="font-size:28px">{icon2} {pred['prob']*100:.1f}%</p>
                <p style="color:#718096;font-size:12px;margin:0">
                    {SNAPSHOT_LABELS.get(snap_name,'').split('—')[1].strip()}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── SHAP do snapshot com maior risco ─────────────────────────────────────
    st.markdown("#### Explicação SHAP")
    snap_sel = st.selectbox(
        "Snapshot para explicação:",
        list(predictions.keys()),
        index=list(predictions.keys()).index(best_snap),
        format_func=lambda s: SNAPSHOT_LABELS.get(s, s),
        key="shap_snap_sel",
    )
    fig_shap = shap_waterfall(predictions[snap_sel], snap_sel)
    if fig_shap:
        st.plotly_chart(fig_shap, use_container_width=True, key="shap_chart")

    st.caption(
        "🔴 Vermelho = aumenta o risco de readmissão  |  "
        "🟢 Verde = reduz o risco de readmissão  |  "
        "Valores SHAP = contribuição marginal de cada feature"
    )

# ── App principal ─────────────────────────────────────────────────────────────

def main():
    # Cabeçalho
    st.markdown("""
    <div class="header-box">
        <h1>🏥 Predição de Readmissão em UTI</h1>
        <p>Sistema de apoio à decisão clínica — MIMIC-IV v3.1 &nbsp;|&nbsp;
           Modelos v7 (Optuna per-snapshot) &nbsp;|&nbsp; TCC IFSC 2026</p>
    </div>
    """, unsafe_allow_html=True)

    bundles   = load_all_bundles()
    all_snaps = load_all_snapshots()

    # Sidebar — seleção de modelo
    with st.sidebar:
        st.markdown("### ⚙️ Configurações")
        model_name = st.selectbox(
            "Modelo",
            ["XGBoost", "LightGBM", "Ensemble", "RandomForest", "LogReg"],
            index=0,
            help="XGBoost D0 é o melhor modelo geral (AUC 0.648)",
        )
        st.markdown("---")
        st.markdown("**Thresholds de risco**")
        st.markdown("🟢 Baixo: < 30%")
        st.markdown("🟡 Médio: 30–55%")
        st.markdown("🔴 Alto: > 55%")
        st.markdown("---")
        st.markdown("**Snapshots disponíveis**")
        for name, label in SNAPSHOT_LABELS.items():
            n = len(all_snaps[name])
            st.markdown(f"- **{name}**: {n:,} stays")

    # Abas principais
    tab1, tab2 = st.tabs(["🔍  Busca por Paciente (stay_id)", "📂  Upload CSV"])

    # ── ABA 1: Busca por stay_id ──────────────────────────────────────────────
    with tab1:
        st.markdown("#### Buscar paciente nos snapshots MIMIC-IV")
        st.caption("Digite o stay_id de qualquer internação em UTI presente nos snapshots gerados.")

        col_input, col_btn = st.columns([3, 1])
        with col_input:
            stay_id_input = st.number_input(
                "stay_id", min_value=1, value=None,
                placeholder="Ex: 30000426",
                label_visibility="collapsed",
            )
        with col_btn:
            buscar = st.button("🔍 Buscar", use_container_width=True, key="btn_buscar")

        if buscar and stay_id_input:
            stay_id = int(stay_id_input)
            patient_snaps = {
                name: df[df["stay_id"] == stay_id].copy()
                for name, df in all_snaps.items()
                if not df[df["stay_id"] == stay_id].empty
            }
            if not patient_snaps:
                st.error(f"stay_id **{stay_id}** não encontrado em nenhum snapshot.")
            else:
                snaps_found = list(patient_snaps.keys())
                st.success(f"Paciente encontrado em: {', '.join(snaps_found)}")

                # Info do paciente
                row = list(patient_snaps.values())[0].iloc[0]
                info_cols = st.columns(4)
                fields = [
                    ("stay_id", "Stay ID"),
                    ("subject_id", "Patient ID"),
                    ("age", "Idade"),
                    ("first_careunit", "Unidade"),
                ]
                for col, (field, label) in zip(info_cols, fields):
                    val = row.get(field, "—")
                    col.metric(label, val if pd.notna(val) else "—")

                predict_and_display(patient_snaps, bundles, model_name)

        elif buscar and not stay_id_input:
            st.warning("Digite um stay_id válido.")

    # ── ABA 2: Upload CSV ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Dados do paciente via CSV")

        # Template download
        col_info, col_dl = st.columns([3, 1])
        with col_info:
            st.caption(
                "Faça upload de um arquivo CSV ou cole o conteúdo diretamente. "
                "Baixe o template para ver as colunas esperadas."
            )
        with col_dl:
            snap_tmpl = st.selectbox(
                "Snapshot do template",
                SNAPSHOT_NAMES, index=0,
                label_visibility="collapsed",
                key="tmpl_snap",
            )
            template_df = get_template_csv(snap_tmpl)
            buf = io.BytesIO()
            template_df.to_csv(buf, index=False)
            st.download_button(
                "📥 Template CSV",
                buf.getvalue(),
                file_name=f"template_{snap_tmpl.lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Modo de entrada
        input_mode = st.radio(
            "Modo de entrada:",
            ["📂 Upload de arquivo", "📋 Colar como texto"],
            horizontal=True,
            key="csv_input_mode",
        )

        df_up = None

        if input_mode == "📂 Upload de arquivo":
            uploaded = st.file_uploader(
                "Arraste ou selecione o CSV",
                type=["csv"],
                help="Uma linha por paciente/snapshot. stay_id é obrigatório.",
                key="csv_upload",
            )
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    st.success(f"Arquivo carregado: {len(df_up)} linha(s), {df_up.shape[1]} colunas")
                except Exception as e:
                    st.error(f"Erro ao ler arquivo: {e}")

        else:  # Colar como texto
            st.caption("Cole o conteúdo do CSV abaixo (inclua o cabeçalho):")
            csv_text = st.text_area(
                "Conteúdo CSV",
                height=200,
                placeholder="stay_id,age,heart_rate_mean,...\n38292992,65,88.3,...",
                label_visibility="collapsed",
                key="csv_paste",
            )
            if csv_text.strip():
                try:
                    df_up = pd.read_csv(io.StringIO(csv_text))
                    st.success(f"CSV reconhecido: {len(df_up)} linha(s), {df_up.shape[1]} colunas")
                except Exception as e:
                    st.error(f"Erro ao interpretar o texto: {e}")

        # Processamento comum
        if df_up is not None:
            snap_choice = st.selectbox(
                "Qual snapshot corresponde a estes dados?",
                SNAPSHOT_NAMES,
                format_func=lambda s: SNAPSHOT_LABELS[s],
                key="csv_snap_choice",
            )

            if "stay_id" not in df_up.columns:
                df_up.insert(0, "stay_id", range(1, len(df_up) + 1))

            if len(df_up) > 1:
                st.info(f"{len(df_up)} pacientes. Selecione qual analisar:")
                row_idx = st.selectbox(
                    "Paciente:",
                    df_up.index,
                    format_func=lambda i: f"Linha {i+1} — stay_id: {df_up.loc[i, 'stay_id']}",
                    key="csv_row_sel",
                )
                df_row = df_up.loc[[row_idx]].copy()
            else:
                df_row = df_up.copy()

            if st.button("▶️ Calcular Risco", use_container_width=False, key="btn_calc"):
                predict_and_display({snap_choice: df_row}, bundles, model_name)


if __name__ == "__main__":
    main()
