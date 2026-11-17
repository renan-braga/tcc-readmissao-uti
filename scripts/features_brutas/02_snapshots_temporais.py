"""
Snapshots Temporais — Sistema de Previsão de Readmissão em UTI
MIMIC-IV v3.1

Gera snapshots cumulativos D0–D3 por internação elegível.
Critérios de exclusão, readmissão em 72h e agregações SOFA corrigidos.

Autor: Renan de Almeida Simões Braga
"""

import pandas as pd
import numpy as np
import psycopg2
import warnings
warnings.filterwarnings('ignore')

# ── Conexão ──────────────────────────────────────────────────────────────────

DB_PARAMS = dict(host="localhost", port="5432", database="postgres", user="renanbraga")
OUTPUT_DIR = "/Users/renanbraga/tcc_ccai/scripts"

def get_conn():
    return psycopg2.connect(**DB_PARAMS)

def query(sql):
    with get_conn() as conn:
        return pd.read_sql(sql, conn)

# ── Configuração dos snapshots ────────────────────────────────────────────────

# Extração CUMULATIVA: cada snapshot usa dados de 0h até end_hours
SNAPSHOTS = {
    "D0": {"end_hours": 24,  "min_los": 24,  "label": "Admissão (0–24h)"},
    "D1": {"end_hours": 48,  "min_los": 48,  "label": "Estabilização (0–48h)"},
    "D2": {"end_hours": 72,  "min_los": 72,  "label": "Evolução (0–72h)"},
    "D3": {"end_hours": 96,  "min_los": 96,  "label": "Consolidação (0–96h)"},
}

# ── Construção da coorte elegível ─────────────────────────────────────────────

COHORT_SQL = """
WITH base AS (
    SELECT
        i.stay_id,
        i.subject_id,
        i.hadm_id,
        i.intime,
        i.outtime,
        i.first_careunit,
        i.last_careunit,
        EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0 AS icu_los_hours,
        p.anchor_age,
        p.anchor_year,
        p.anchor_year_group,
        p.gender,
        a.admittime,
        a.dischtime,
        a.admission_type,
        a.admission_location,
        a.discharge_location,
        a.deathtime
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
    JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
    WHERE i.outtime IS NOT NULL
),

eligible AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS stay_rank
    FROM base
    WHERE anchor_age >= 18                                         -- E8: adultos
      AND icu_los_hours >= 4                                       -- E7: >= 4h
      AND (deathtime IS NULL OR deathtime > outtime)              -- E1: não morreu na UTI
      AND discharge_location NOT ILIKE '%OTHER HOSP%'            -- E5: sem transferência
      AND discharge_location NOT ILIKE '%TRANSFER%'
      AND discharge_location NOT ILIKE '%HOSPICE%'               -- E6: sem paliativos
),

next_icu AS (
    -- Próxima internação em UTI dentro da mesma hospitalização (mesmo hadm_id)
    SELECT
        e.stay_id,
        MIN(i2.intime) AS next_icu_intime
    FROM eligible e
    JOIN mimiciv_icu.icustays i2
        ON  e.subject_id = i2.subject_id
        AND i2.hadm_id   = e.hadm_id
        AND i2.intime    > e.outtime
        AND EXTRACT(EPOCH FROM (i2.intime - e.outtime)) / 3600.0 >= 4  -- E4: gap >= 4h (não transferência)
    GROUP BY e.stay_id
)

SELECT
    e.stay_id,
    e.subject_id,
    e.hadm_id,
    e.stay_rank,
    e.intime,
    e.outtime,
    e.icu_los_hours,
    e.anchor_age    AS age,
    e.anchor_year,
    e.anchor_year_group,
    e.gender,
    e.first_careunit,
    e.last_careunit,
    e.admission_type,
    e.admission_location,
    e.discharge_location,
    e.deathtime,
    ni.next_icu_intime,
    EXTRACT(EPOCH FROM (ni.next_icu_intime - e.outtime)) / 3600.0 AS hours_until_readmission,
    CASE
        WHEN ni.next_icu_intime IS NOT NULL
         AND EXTRACT(EPOCH FROM (ni.next_icu_intime - e.outtime)) / 3600.0 <= 72
        THEN 1 ELSE 0
    END AS readmitted
FROM eligible e
LEFT JOIN next_icu ni ON e.stay_id = ni.stay_id
-- E2: excluir pacientes que morreram na enfermaria em até 72h SEM readmissão
WHERE NOT (
    e.deathtime IS NOT NULL
    AND e.deathtime > e.outtime
    AND e.deathtime <= (e.outtime + INTERVAL '72 hours')
    AND (ni.next_icu_intime IS NULL OR ni.next_icu_intime >= e.deathtime)
)
ORDER BY e.subject_id, e.intime
"""

# ── Extração de dados clínicos (cumulativa 0 → end_hours) ────────────────────

def get_chartevents(end_hours):
    """Vitais, GCS e FiO2 — cumulativo de 0 a end_hours."""
    sql = f"""
    SELECT
        ce.stay_id,
        ce.itemid,
        ce.valuenum
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
    WHERE ce.itemid IN (
        220045,  -- Heart Rate
        220050,  -- Systolic BP
        220051,  -- Diastolic BP
        220052,  -- MAP
        220210,  -- Respiratory Rate
        223761,  -- Temperature F
        223762,  -- Temperature C
        220277,  -- SpO2
        220739,  -- GCS Eye
        223900,  -- GCS Verbal
        223901,  -- GCS Motor
        223835   -- FiO2
    )
    AND ce.valuenum IS NOT NULL
    AND ce.valuenum > 0
    AND ce.charttime BETWEEN ie.intime
                         AND (ie.intime + INTERVAL '{end_hours} hours')
    """
    return query(sql)


def get_labevents(end_hours):
    """Labs — cumulativo 0 a end_hours, filtrado pelo período da UTI (via stay_id)."""
    sql = f"""
    SELECT
        ie.stay_id,
        le.itemid,
        le.valuenum
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays ie
        ON  le.hadm_id   = ie.hadm_id
        AND le.charttime BETWEEN ie.intime
                             AND (ie.intime + INTERVAL '{end_hours} hours')
    WHERE le.itemid IN (
        51265,  -- Platelets
        50885,  -- Bilirubin Total
        50912,  -- Creatinine
        50821,  -- PO2 (gasometria arterial)
        50818,  -- PCO2
        50820,  -- pH
        50983,  -- Sodium
        50971,  -- Potassium
        51006,  -- Urea Nitrogen (BUN)
        51301,  -- WBC
        50811,  -- Hemoglobin
        50810,  -- Hematocrit
        50882   -- Bicarbonate
    )
    AND le.valuenum IS NOT NULL
    """
    return query(sql)


def get_urine_output(end_hours):
    """Débito urinário — soma cumulativa 0 a end_hours."""
    sql = f"""
    SELECT
        oe.stay_id,
        SUM(oe.value) AS urine_output_ml
    FROM mimiciv_icu.outputevents oe
    JOIN mimiciv_icu.icustays ie ON oe.stay_id = ie.stay_id
    WHERE oe.itemid IN (
        226559, 226560, 226561, 226584, 226563,
        226564, 226565, 226567, 226557, 226558
    )
    AND oe.value IS NOT NULL
    AND oe.value > 0
    AND oe.charttime BETWEEN ie.intime
                         AND (ie.intime + INTERVAL '{end_hours} hours')
    GROUP BY oe.stay_id
    """
    return query(sql)


def get_vasopressors(end_hours):
    """Vasopressores — dose máxima no período 0 a end_hours."""
    sql = f"""
    SELECT
        ie.stay_id,
        MAX(CASE WHEN ie.itemid = 221906 THEN ie.rate END) AS norepinephrine_max,
        MAX(CASE WHEN ie.itemid = 221289 THEN ie.rate END) AS epinephrine_max,
        MAX(CASE WHEN ie.itemid = 221662 THEN ie.rate END) AS dopamine_max,
        MAX(CASE WHEN ie.itemid = 221653 THEN ie.rate END) AS dobutamine_max,
        MAX(CASE WHEN ie.itemid IN (221906,221289,221662,221653) THEN ie.rate END) AS any_vasopressor_max
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.itemid IN (221906, 221289, 221662, 221653)
      AND ie.rate IS NOT NULL
      AND ie.rate > 0
      AND ie.starttime BETWEEN icu.intime
                           AND (icu.intime + INTERVAL '{end_hours} hours')
    GROUP BY ie.stay_id
    """
    return query(sql)


def get_derived_scores(_end_hours):
    """OASIS e Charlson do schema mimiciv_derived (pré-computados)."""
    oasis_sql = """
    SELECT stay_id, oasis
    FROM mimiciv_derived.oasis
    """
    charlson_sql = """
    SELECT hadm_id, charlson_comorbidity_index AS cci
    FROM mimiciv_derived.charlson
    """
    try:
        df_oasis = query(oasis_sql)
    except Exception:
        df_oasis = pd.DataFrame(columns=["stay_id", "oasis"])
    try:
        df_charlson = query(charlson_sql)
    except Exception:
        df_charlson = pd.DataFrame(columns=["hadm_id", "cci"])
    return df_oasis, df_charlson


def get_sofa_pointintime(end_hours):
    """Extrai SOFA e labs críticos no instante mais próximo do cutoff do snapshot.

    mimiciv_derived.sofa.sofa_24hours é uma janela ROLANTE de 24h (não cumulativa),
    portanto o delta entre snapshots consecutivos é sinal real de trajetória clínica:
      delta_sofa_pt(D1) = sofa_pt(48h) - sofa_pt(24h) → negativo = melhora, positivo = piora.

    Seleciona o hr mais alto ≤ end_hours para cada stay, capturando o estado clínico
    mais recente disponível antes do cutoff.
    """
    sql = f"""
    WITH ranked AS (
        SELECT
            stay_id,
            sofa_24hours,
            respiration_24hours,
            coagulation_24hours,
            liver_24hours,
            cardiovascular_24hours,
            cns_24hours,
            renal_24hours,
            gcs_min,
            creatinine_max,
            platelet_min,
            bilirubin_max,
            meanbp_min,
            uo_24hr,
            COALESCE(pao2fio2ratio_vent, pao2fio2ratio_novent) AS pao2fio2_ratio,
            ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY hr DESC) AS rn
        FROM mimiciv_derived.sofa
        WHERE hr <= {end_hours}
    )
    SELECT
        stay_id,
        sofa_24hours            AS sofa_pt,
        respiration_24hours     AS sofa_pt_respiration,
        coagulation_24hours     AS sofa_pt_coagulation,
        liver_24hours           AS sofa_pt_liver,
        cardiovascular_24hours  AS sofa_pt_cardiovascular,
        cns_24hours             AS sofa_pt_neurologic,
        renal_24hours           AS sofa_pt_renal,
        gcs_min                 AS gcs_pt,
        creatinine_max          AS creatinine_pt,
        platelet_min            AS platelet_pt,
        bilirubin_max           AS bilirubin_pt,
        meanbp_min              AS meanbp_pt,
        uo_24hr                 AS urine_output_pt,
        pao2fio2_ratio          AS pao2fio2_pt
    FROM ranked
    WHERE rn = 1
    """
    try:
        return query(sql)
    except Exception:
        return pd.DataFrame(columns=["stay_id", "sofa_pt"])


# ── Processamento dos dados clínicos ─────────────────────────────────────────

CHART_MAP = {
    220045: "heart_rate",
    220050: "systolic_bp",
    220051: "diastolic_bp",
    220052: "map",
    220210: "respiratory_rate",
    223761: "temperature_f",
    223762: "temperature_c",
    220277: "spo2",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
    223835: "fio2",
}

LAB_MAP = {
    51265: "platelets",
    50885: "bilirubin",
    50912: "creatinine",
    50821: "po2",
    50818: "pco2",
    50820: "ph",
    50983: "sodium",
    50971: "potassium",
    51006: "bun",
    51301: "wbc",
    50811: "hemoglobin",
    50810: "hematocrit",
    50882: "bicarbonate",
}

# Agregações: por variável, qual estatística usar
# "worst" = MAX para variáveis onde valor alto é ruim, MIN onde valor baixo é ruim
CHART_AGG = {
    "heart_rate":       {"mean": True, "min": True, "max": True},
    "systolic_bp":      {"mean": True, "min": True},       # min para hipotensão
    "diastolic_bp":     {"mean": True},
    "map":              {"mean": True},
    "respiratory_rate": {"mean": True, "max": True},
    "temperature_f":    {"mean": True},
    "temperature_c":    {"mean": True},
    "spo2":             {"mean": True, "min": True},        # min para hipoxemia
    "gcs_eye":          {"min": True},                      # min = pior GCS
    "gcs_verbal":       {"min": True},
    "gcs_motor":        {"min": True},
    "fio2":             {"mean": True, "max": True},        # max para SOFA respiratório
}

LAB_AGG = {
    "platelets":   {"min": True, "mean": True},             # min = pior SOFA coagulação
    "bilirubin":   {"max": True, "mean": True},             # max = pior SOFA hepático
    "creatinine":  {"max": True, "mean": True},             # max = pior SOFA renal
    "po2":         {"min": True, "mean": True},             # min = pior oxigenação
    "pco2":        {"mean": True},
    "ph":          {"min": True, "mean": True},
    "sodium":      {"mean": True},
    "potassium":   {"mean": True, "max": True},
    "bun":         {"max": True, "mean": True},
    "wbc":         {"mean": True, "max": True},
    "hemoglobin":  {"min": True, "mean": True},
    "hematocrit":  {"min": True, "mean": True},
    "bicarbonate": {"min": True, "mean": True},
}


def aggregate_chart(df_raw):
    df_raw["param"] = df_raw["itemid"].map(CHART_MAP)
    df_raw = df_raw.dropna(subset=["param"])

    rows = []
    for (stay_id, param), grp in df_raw.groupby(["stay_id", "param"]):
        agg_spec = CHART_AGG.get(param, {"mean": True})
        row = {"stay_id": stay_id}
        vals = grp["valuenum"]
        if agg_spec.get("mean"):
            row[f"{param}_mean"] = vals.mean()
        if agg_spec.get("min"):
            row[f"{param}_min"] = vals.min()
        if agg_spec.get("max"):
            row[f"{param}_max"] = vals.max()
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("stay_id").first().reset_index()


def aggregate_labs(df_raw):
    df_raw["param"] = df_raw["itemid"].map(LAB_MAP)
    df_raw = df_raw.dropna(subset=["param"])

    rows = []
    for (stay_id, param), grp in df_raw.groupby(["stay_id", "param"]):
        agg_spec = LAB_AGG.get(param, {"mean": True})
        row = {"stay_id": stay_id}
        vals = grp["valuenum"]
        if agg_spec.get("mean"):
            row[f"{param}_mean"] = vals.mean()
        if agg_spec.get("min"):
            row[f"{param}_min"] = vals.min()
        if agg_spec.get("max"):
            row[f"{param}_max"] = vals.max()
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("stay_id").first().reset_index()

# ── Scores SOFA ──────────────────────────────────────────────────────────────

def sofa_platelets(v):
    if pd.isna(v): return np.nan
    if v >= 150: return 0
    if v >= 100: return 1
    if v >= 50:  return 2
    if v >= 20:  return 3
    return 4

def sofa_bilirubin(v):
    if pd.isna(v): return np.nan
    if v < 1.2:  return 0
    if v < 2.0:  return 1
    if v < 6.0:  return 2
    if v < 12.0: return 3
    return 4

def sofa_creatinine(v):
    if pd.isna(v): return np.nan
    if v < 1.2:  return 0
    if v < 2.0:  return 1
    if v < 3.5:  return 2
    if v < 5.0:  return 3
    return 4

def sofa_gcs(v):
    if pd.isna(v): return np.nan
    if v >= 15: return 0
    if v >= 13: return 1
    if v >= 10: return 2
    if v >= 6:  return 3
    return 4

def sofa_cardiovascular(map_val, vaso_max):
    if pd.isna(map_val): return np.nan
    if pd.isna(vaso_max) or vaso_max == 0:
        return 0 if map_val >= 70 else 1
    if vaso_max < 5:  return 2
    if vaso_max < 15: return 3
    return 4

def sofa_respiratory(po2_min, fio2_max):
    """Usa pior PO2 (min) com maior FiO2 registrada. Fallback: FiO2=0.21."""
    if pd.isna(po2_min): return np.nan
    fio2 = fio2_max if (not pd.isna(fio2_max) and fio2_max > 0) else 0.21
    # FiO2 pode vir em % (21–100) ou fração (0.21–1.0)
    if fio2 > 1:
        fio2 = fio2 / 100.0
    ratio = po2_min / fio2
    if ratio >= 400: return 0
    if ratio >= 300: return 1
    if ratio >= 200: return 2
    if ratio >= 100: return 3
    return 4


def calculate_sofa(df):
    if "platelets_min" in df.columns:
        df["sofa_coagulation"] = df["platelets_min"].apply(sofa_platelets)
    if "bilirubin_max" in df.columns:
        df["sofa_liver"] = df["bilirubin_max"].apply(sofa_bilirubin)
    if "creatinine_max" in df.columns:
        df["sofa_renal"] = df["creatinine_max"].apply(sofa_creatinine)

    # GCS total = soma dos mínimos de cada componente (pior estado)
    gcs_cols = [c for c in ["gcs_eye_min", "gcs_verbal_min", "gcs_motor_min"] if c in df.columns]
    if len(gcs_cols) == 3:
        df["gcs_total_min"] = df[gcs_cols].sum(axis=1, skipna=False)
        df["sofa_neurologic"] = df["gcs_total_min"].apply(sofa_gcs)

    if "map_mean" in df.columns:
        vaso = df.get("any_vasopressor_max", pd.Series(np.nan, index=df.index))
        df["sofa_cardiovascular"] = df.apply(
            lambda r: sofa_cardiovascular(r["map_mean"], vaso.get(r.name, np.nan)), axis=1
        )

    if "po2_min" in df.columns:
        fio2 = df.get("fio2_max", pd.Series(np.nan, index=df.index))
        df["sofa_respiratory"] = df.apply(
            lambda r: sofa_respiratory(r["po2_min"], fio2.get(r.name, np.nan)), axis=1
        )

    sofa_cols = [c for c in ["sofa_coagulation", "sofa_liver", "sofa_renal",
                               "sofa_neurologic", "sofa_cardiovascular", "sofa_respiratory"]
                 if c in df.columns]
    if sofa_cols:
        df["sofa_total"] = df[sofa_cols].sum(axis=1, skipna=True)

    return df

# ── Pipeline principal por snapshot ──────────────────────────────────────────

def build_snapshot(snapshot_id, config, cohort, df_oasis, df_charlson):
    end_h = config["end_hours"]
    min_los = config["min_los"]
    print(f"\n{'='*60}")
    print(f"  Snapshot {snapshot_id} — {config['label']}")
    print(f"  Extração cumulativa: 0–{end_h}h | LOS mínimo: {min_los}h")
    print(f"{'='*60}")

    # Filtrar pacientes com LOS suficiente para este snapshot
    base = cohort[cohort["icu_los_hours"] >= min_los].copy()
    print(f"  Stays elegíveis: {len(base)} (de {len(cohort)} na coorte)")
    print(f"  Taxa readmissão: {base['readmitted'].mean():.3f} ({base['readmitted'].mean()*100:.1f}%)")

    # Extrair dados clínicos
    print("  Extraindo chartevents...")
    df_chart_raw = get_chartevents(end_h)
    print(f"    {len(df_chart_raw):,} registros")

    print("  Extraindo labevents...")
    df_lab_raw = get_labevents(end_h)
    print(f"    {len(df_lab_raw):,} registros")

    print("  Extraindo urine output...")
    df_urine = get_urine_output(end_h)

    print("  Extraindo vasopressores...")
    df_vaso = get_vasopressors(end_h)

    # Filtrar para stays da coorte
    stay_ids = set(base["stay_id"])
    df_chart_raw = df_chart_raw[df_chart_raw["stay_id"].isin(stay_ids)]
    df_lab_raw   = df_lab_raw[df_lab_raw["stay_id"].isin(stay_ids)]
    df_urine     = df_urine[df_urine["stay_id"].isin(stay_ids)] if len(df_urine) else df_urine
    df_vaso      = df_vaso[df_vaso["stay_id"].isin(stay_ids)] if len(df_vaso) else df_vaso

    # Agregar
    df_chart = aggregate_chart(df_chart_raw)
    df_lab   = aggregate_labs(df_lab_raw)

    # Montar dataset
    df = base.copy()
    for aux in [df_chart, df_lab, df_urine, df_vaso]:
        if len(aux) > 0:
            df = df.merge(aux, on="stay_id", how="left")

    # Scores derivados (OASIS, Charlson)
    if len(df_oasis) > 0:
        df = df.merge(df_oasis, on="stay_id", how="left")
    if len(df_charlson) > 0:
        df = df.merge(df_charlson, on="hadm_id", how="left")

    # SOFA cumulativo (derivado das labs agregadas)
    df = calculate_sofa(df)

    # SOFA point-in-time (janela rolante 24h de mimiciv_derived.sofa)
    # Delta entre snapshots captura trajetória real (melhora/piora)
    print("  Extraindo SOFA point-in-time...")
    df_sofa_pt = get_sofa_pointintime(end_h)
    df_sofa_pt = df_sofa_pt[df_sofa_pt["stay_id"].isin(stay_ids)]
    if len(df_sofa_pt) > 0:
        df = df.merge(df_sofa_pt, on="stay_id", how="left")
        n_valid = int(df["sofa_pt"].notna().sum())
        print(f"    {n_valid:,} stays com SOFA point-in-time ({n_valid/len(df)*100:.1f}%)")

    # Temperatura: unificar F e C
    if "temperature_f_mean" in df.columns and "temperature_c_mean" in df.columns:
        df["temperature_c_mean"] = df["temperature_c_mean"].fillna(
            (df["temperature_f_mean"] - 32) * 5 / 9
        )
        df = df.drop(columns=["temperature_f_mean"], errors="ignore")

    print(f"  Dataset final: {df.shape[0]} linhas × {df.shape[1]} colunas")

    out_path = f"{OUTPUT_DIR}/snapshot_{snapshot_id.lower()}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Salvo: {out_path}")

    return df


def main():
    print("SNAPSHOTS TEMPORAIS — MIMIC-IV v3.1")
    print("=" * 60)
    print("Definição de readmissão: retorno à UTI em ≤72h (mesma internação)")
    print("Extração: cumulativa (0 → end_hours)")
    print("=" * 60)

    # 1. Coorte elegível
    print("\nConstruindo coorte elegível...")
    cohort = query(COHORT_SQL)
    print(f"  Total de stays elegíveis: {len(cohort)}")
    print(f"  Pacientes únicos: {cohort['subject_id'].nunique()}")
    print(f"  Taxa global de readmissão (72h): {cohort['readmitted'].mean():.3f} "
          f"({cohort['readmitted'].mean()*100:.1f}%)")
    print(f"  Readmissões identificadas: {cohort['readmitted'].sum()}")

    # 2. Scores pré-computados
    print("\nCarregando scores derivados (OASIS, Charlson)...")
    df_oasis, df_charlson = get_derived_scores(0)  # arg ignorado (tabelas completas)
    print(f"  OASIS: {len(df_oasis)} registros")
    print(f"  Charlson: {len(df_charlson)} registros")

    # 3. Gerar snapshots
    summaries = {}
    for sid, cfg in SNAPSHOTS.items():
        try:
            df_snap = build_snapshot(sid, cfg, cohort, df_oasis, df_charlson)
            summaries[sid] = {
                "stays": len(df_snap),
                "readmission_rate": df_snap["readmitted"].mean(),
                "columns": df_snap.shape[1],
            }
        except Exception as e:
            import traceback
            print(f"  ERRO no snapshot {sid}: {e}")
            traceback.print_exc()
            summaries[sid] = None

    # 4. Resumo
    print("\n" + "=" * 60)
    print("RESUMO DOS SNAPSHOTS")
    print("=" * 60)
    for sid, s in summaries.items():
        if s:
            print(f"  {sid}: {s['stays']:,} stays | "
                  f"readmissão {s['readmission_rate']:.3f} ({s['readmission_rate']*100:.1f}%) | "
                  f"{s['columns']} colunas")
        else:
            print(f"  {sid}: FALHOU")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
