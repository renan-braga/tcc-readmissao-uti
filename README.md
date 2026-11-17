# ICU Readmission Prediction with MIMIC-IV

Undergraduate thesis (TCC). It is a case study of a modular architecture that predicts **ICU readmission within 72 hours** of discharge, using the MIMIC-IV v3.1 database.

## Summary

- **Temporal snapshots.** Four independent classifiers are trained on cumulative 24-hour windows (D0 to D3). Each one only sees the data available up to its cutoff. This adapts a snapshot strategy originally designed for emergency departments.
- **Cohort.** 65,760 ICU stays from 48,524 patients, with a 4.5% readmission rate.
- **Models.** Logistic Regression, Random Forest, XGBoost, LightGBM and an ensemble. Hyperparameters are tuned with Optuna, and train/test splits are made at the patient level.
- **Leakage audit.** Variables only known at discharge (such as total length of stay) were removed.
- **Best model.** XGBoost on D2: AUC-ROC 0.644, sensitivity 0.62, precision 0.083 at the chosen operating point.
- **Explainability (SHAP).** Chronic comorbidity burden dominates over acute physiology. Feature importance shifts across time cutoffs, which supports the snapshot design.

The full paper (in Portuguese) is in [`template/tcc-renan.pdf`](template/tcc-renan.pdf), with its LaTeX source in `template/`.

## Data access

MIMIC-IV is a credentialed dataset distributed by [PhysioNet](https://physionet.org/content/mimiciv/). Under its data use agreement, **no data, derived patient-level files or trained models are included in this repository**. To reproduce the results:

1. Get credentialed access to MIMIC-IV v3.1 on PhysioNet.
2. Load it into PostgreSQL.
3. Build the derived concepts with [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code). The scripts use `mimiciv_derived.sofa`, `oasis`, `apsiii`, `sapsii`, `charlson` and `kdigo_creatinine`.

## Repository layout

```
scripts/
  features_brutas/   Main pipeline with raw clinical features
    02_snapshots_temporais.py      build D0–D3 snapshots from PostgreSQL
    03_treinamento_xgboost.py      baseline training + SHAP
    04_tuning_optuna.py            Optuna tuning (LightGBM, XGBoost)
    05_tuning_v7_per_snapshot.py   per-snapshot tuning (final model)
    06_trajectory_model.py         D0→D3 trajectory model
    07_shap_v7.py                  SHAP for the final model
    08_curvas_roc_pt.py            ROC / precision-recall figures
  scores_lean/       Variant using only validated clinical scores
                     (OASIS, CCI, APSIII, SAPSII, SOFA, KDIGO)
  02_*.py, 03_*.py   Same as steps 02–03 in features_brutas/
app/                 Streamlit demo for per-patient predictions with SHAP
template/            Paper source (IEEE Access template) and figures
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt lightgbm optuna streamlit
```

Run the scripts in numeric order. Database connection settings and output folders are defined as constants at the top of each script.

To run the demo app, you need the generated snapshots and trained models. Point `MODELS_DIR` and `SNAPSHOTS_DIR` in `app/app.py` and `app/utils.py` at them, then start it:

```bash
streamlit run app/app.py
```
