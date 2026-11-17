"""
Snapshots Lean — Abordagem por Scores Clínicos Validados
MIMIC-IV v3.1

Extrai apenas scores clínicos pré-computados e features de contexto,
sem variáveis brutas de chartevents/labevents. Objetivo: reduzir overfitting
em D2/D3 por limitação de amostras.

Features por snapshot (~27):
  Estáticos (1x por internação): OASIS, CCI, APSIII, SAPSII
  Point-in-time por snapshot (14): sofa_pt + 6 componentes + 7 labs derivados
  Trajetória por snapshot (1): KDIGO stage
  Contexto: age, gender, admission_type, first_careunit, icu_los_hours, vasopressor_any

Autor: Renan de Almeida Simões Braga
"""

import pandas as pd
import numpy as np
import psycopg2
import warnings
warnings.filterwarnings('ignore')

DB_PARAMS  = dict(host="localhost", port="5432", database="postgres", user="renanbraga")
OUTPUT_DIR = "/Users/renanbraga/tcc_ccai/scripts/scores_lean"

def get_conn():
    return psycopg2.connect(**DB_PARAMS)

def query(sql):
    with get_conn() as conn:
        return pd.read_sql(sql, conn)

SNAPSHOTS = {
    "D0": {"end_hours": 24, "min_los": 24, "label": "Admissão (0–24h)"},
    "D1": {"end_hours": 48, "min_los": 48, "label": "Estabilização (0–48h)"},
    "D2": {"end_hours": 72, "min_los": 72, "label": "Evolução (0–72h)"},
    "D3": {"end_hours": 96, "min_los": 96, "label": "Consolidação (0–96h)"},
}

# ── Coorte (idêntica à abordagem de features brutas) ──────────────────────────

COHORT_SQL = """
WITH base AS (
    SELECT
        i.stay_id, i.subject_id, i.hadm_id,
        i.intime, i.outtime,
        i.first_careunit, i.last_careunit,
        EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0 AS icu_los_hours,
        p.anchor_age AS age,
        p.anchor_year, p.anchor_year_group,
        p.gender,
        a.admittime, a.dischtime,
        a.admission_type, a.admission_location, a.discharge_location,
        a.deathtime,
        ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) AS stay_rank
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
    JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
),
eligible AS (
    SELECT * FROM base
    WHERE icu_los_hours >= 24
      AND age >= 18
),
next_icu AS (
    SELECT e.stay_id, MIN(i2.intime) AS next_icu_intime
    FROM eligible e
    JOIN mimiciv_icu.icustays i2
        ON e.subject_id = i2.subject_id
       AND i2.hadm_id = e.hadm_id
       AND i2.intime > e.outtime
       AND EXTRACT(EPOCH FROM (i2.intime - e.outtime)) / 3600.0 >= 4
    GROUP BY e.stay_id
)
SELECT e.stay_id, e.subject_id, e.hadm_id, e.stay_rank,
       e.intime, e.outtime, e.icu_los_hours,
       e.age, e.anchor_year, e.anchor_year_group, e.gender,
       e.first_careunit, e.last_careunit,
       e.admission_type, e.admission_location, e.discharge_location,
       e.admittime, e.dischtime, e.deathtime,
       ni.next_icu_intime,
       CASE
           WHEN ni.next_icu_intime IS NOT NULL
                AND EXTRACT(EPOCH FROM (ni.next_icu_intime - e.outtime)) / 3600.0 <= 72
           THEN 1 ELSE 0
       END AS readmitted,
       CASE
           WHEN ni.next_icu_intime IS NOT NULL
           THEN EXTRACT(EPOCH FROM (ni.next_icu_intime - e.outtime)) / 3600.0
       END AS hours_until_readmission
FROM eligible e
LEFT JOIN next_icu ni ON e.stay_id = ni.stay_id
WHERE NOT (
    e.deathtime IS NOT NULL
    AND e.deathtime > e.outtime
    AND e.deathtime <= (e.outtime + INTERVAL '72 hours')
    AND (ni.next_icu_intime IS NULL OR ni.next_icu_intime >= e.deathtime)
)
ORDER BY e.subject_id, e.intime
"""

# ── Scores estáticos (primeiro dia) ──────────────────────────────────────────

def get_static_scores():
    """OASIS, CCI, APSIII, SAPSII — scores pré-computados do primeiro dia."""
    oasis_sql   = "SELECT stay_id, oasis FROM mimiciv_derived.oasis"
    charlson_sql= "SELECT hadm_id, charlson_comorbidity_index AS cci FROM mimiciv_derived.charlson"
    apsiii_sql  = "SELECT stay_id, apsiii, apsiii_prob FROM mimiciv_derived.apsiii"
    sapsii_sql  = "SELECT stay_id, sapsii, sapsii_prob FROM mimiciv_derived.sapsii"

    results = {}
    for name, sql in [("oasis", oasis_sql), ("charlson", charlson_sql),
                      ("apsiii", apsiii_sql), ("sapsii", sapsii_sql)]:
        try:
            results[name] = query(sql)
        except Exception as e:
            print(f"    Aviso: {name} não disponível ({e})")
            results[name] = pd.DataFrame()
    return results

# ── SOFA point-in-time (janela rolante 24h) ───────────────────────────────────

def get_sofa_pointintime(end_hours):
    """SOFA e labs críticos no instante mais próximo do cutoff.

    sofa_24hours é janela ROLANTE de 24h — delta entre snapshots é trajetória real.
    """
    sql = f"""
    WITH ranked AS (
        SELECT
            stay_id,
            sofa_24hours,
            respiration_24hours, coagulation_24hours, liver_24hours,
            cardiovascular_24hours, cns_24hours, renal_24hours,
            gcs_min, creatinine_max, platelet_min, bilirubin_max,
            meanbp_min, uo_24hr,
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
    FROM ranked WHERE rn = 1
    """
    try:
        return query(sql)
    except Exception:
        return pd.DataFrame(columns=["stay_id", "sofa_pt"])

# ── KDIGO stage point-in-time ─────────────────────────────────────────────────

def get_kdigo_stage(end_hours):
    """Estadiamento AKI por KDIGO no cutoff do snapshot."""
    sql = f"""
    WITH ranked AS (
        SELECT
            kc.stay_id,
            kc.creat,
            LEAST(kc.creat_low_past_48hr, kc.creat_low_past_7day) AS creat_baseline,
            ROW_NUMBER() OVER (
                PARTITION BY kc.stay_id
                ORDER BY kc.charttime DESC
            ) AS rn
        FROM mimiciv_derived.kdigo_creatinine kc
        JOIN mimiciv_icu.icustays icu ON kc.stay_id = icu.stay_id
        WHERE kc.charttime <= icu.intime + INTERVAL '{end_hours} hours'
    )
    SELECT
        stay_id,
        CASE
            WHEN creat >= 4.0
                 OR (creat_baseline > 0 AND creat / NULLIF(creat_baseline, 0) >= 3.0) THEN 3
            WHEN creat_baseline > 0 AND creat / NULLIF(creat_baseline, 0) >= 2.0 THEN 2
            WHEN creat_baseline > 0 AND creat / NULLIF(creat_baseline, 0) >= 1.5 THEN 1
            ELSE 0
        END AS kdigo_stage
    FROM ranked WHERE rn = 1
    """
    try:
        return query(sql)
    except Exception:
        return pd.DataFrame(columns=["stay_id", "kdigo_stage"])

# ── Vasopressor (binário) ─────────────────────────────────────────────────────

def get_vasopressor_binary(end_hours):
    """1 se qualquer vasopressor foi usado em 0–end_hours, 0 caso contrário."""
    sql = f"""
    SELECT DISTINCT icu.stay_id, 1 AS vasopressor_any
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.itemid IN (221906, 221289, 221662, 221653)
      AND ie.rate IS NOT NULL AND ie.rate > 0
      AND ie.starttime BETWEEN icu.intime
                           AND (icu.intime + INTERVAL '{end_hours} hours')
    """
    try:
        return query(sql)
    except Exception:
        return pd.DataFrame(columns=["stay_id", "vasopressor_any"])

# ── Pipeline por snapshot ─────────────────────────────────────────────────────

def build_snapshot(snapshot_id, config, cohort, static_scores):
    end_h   = config["end_hours"]
    min_los = config["min_los"]
    print(f"\n{'='*60}")
    print(f"  Snapshot {snapshot_id} — {config['label']}")
    print(f"{'='*60}")

    base = cohort[cohort["icu_los_hours"] >= min_los].copy()
    stay_ids = set(base["stay_id"])
    print(f"  Stays elegíveis: {len(base)} | Readmissão: {base['readmitted'].mean():.3f}")

    df = base.copy()

    # Scores estáticos (primeiro dia)
    if len(static_scores.get("oasis", pd.DataFrame())) > 0:
        df = df.merge(static_scores["oasis"], on="stay_id", how="left")
    if len(static_scores.get("charlson", pd.DataFrame())) > 0:
        df = df.merge(static_scores["charlson"], on="hadm_id", how="left")
    if len(static_scores.get("apsiii", pd.DataFrame())) > 0:
        df = df.merge(static_scores["apsiii"][["stay_id", "apsiii", "apsiii_prob"]], on="stay_id", how="left")
    if len(static_scores.get("sapsii", pd.DataFrame())) > 0:
        df = df.merge(static_scores["sapsii"][["stay_id", "sapsii", "sapsii_prob"]], on="stay_id", how="left")

    # SOFA point-in-time
    print("  Extraindo SOFA point-in-time...")
    df_sofa = get_sofa_pointintime(end_h)
    df_sofa = df_sofa[df_sofa["stay_id"].isin(stay_ids)]
    if len(df_sofa) > 0:
        df = df.merge(df_sofa, on="stay_id", how="left")
        print(f"    {int(df['sofa_pt'].notna().sum()):,} stays com SOFA pt ({int(df['sofa_pt'].notna().sum())/len(df)*100:.1f}%)")

    # KDIGO stage
    print("  Extraindo KDIGO stage...")
    df_kdigo = get_kdigo_stage(end_h)
    df_kdigo = df_kdigo[df_kdigo["stay_id"].isin(stay_ids)]
    if len(df_kdigo) > 0:
        df = df.merge(df_kdigo, on="stay_id", how="left")
        df["kdigo_stage"] = df["kdigo_stage"].fillna(0).astype(int)

    # Vasopressor binário
    print("  Extraindo vasopressores...")
    df_vaso = get_vasopressor_binary(end_h)
    df_vaso = df_vaso[df_vaso["stay_id"].isin(stay_ids)]
    if len(df_vaso) > 0:
        df = df.merge(df_vaso, on="stay_id", how="left")
    df["vasopressor_any"] = df.get("vasopressor_any", pd.Series(0, index=df.index)).fillna(0).astype(int)

    print(f"  Dataset final: {df.shape[0]} linhas × {df.shape[1]} colunas")
    out = f"{OUTPUT_DIR}/snapshot_{snapshot_id.lower()}.csv"
    df.to_csv(out, index=False)
    print(f"  Salvo: {out}")
    return df


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("SNAPSHOTS SCORES LEAN — MIMIC-IV v3.1")
    print("=" * 60)

    print("\nConstruindo coorte...")
    cohort = query(COHORT_SQL)
    print(f"  {len(cohort):,} stays | {cohort['subject_id'].nunique():,} pacientes | readmissão {cohort['readmitted'].mean():.3f}")

    print("\nCarregando scores estáticos (OASIS, CCI, APSIII, SAPSII)...")
    static_scores = get_static_scores()
    for name, df in static_scores.items():
        print(f"  {name}: {len(df):,} registros")

    summaries = {}
    for sid, cfg in SNAPSHOTS.items():
        try:
            df_snap = build_snapshot(sid, cfg, cohort, static_scores)
            summaries[sid] = {
                "stays": len(df_snap),
                "readmission_rate": df_snap["readmitted"].mean(),
                "columns": df_snap.shape[1],
            }
        except Exception as e:
            import traceback
            print(f"ERRO no snapshot {sid}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("RESUMO")
    print("=" * 60)
    for sid, s in summaries.items():
        print(f"  {sid}: {s['stays']:,} stays | readmissão {s['readmission_rate']:.3f} ({s['readmission_rate']*100:.1f}%) | {s['columns']} colunas")


if __name__ == "__main__":
    main()
