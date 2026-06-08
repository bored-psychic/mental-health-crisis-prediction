# 🧠 Mental Health Crisis Prediction

**Predicting a PHQ-9 depression crisis (score ≥ 10) 48–72 hours in advance from
passive smartphone + wearable sensing data.**

A reproducible, end-to-end machine-learning pipeline that compares five models
(Logistic Regression, Decision Tree, SVM, Random Forest, **XGBoost**) on
time-lagged behavioural features, explains them with **SHAP**, and validates
generalisation across two datasets (GLOBEM → StudentLife).

> ⚠️ **Decision-support prototype only — not a diagnostic or treatment tool.**
> All predictions must be reviewed by a qualified clinician.

---

## Quick start

The project **runs immediately on synthetic GLOBEM-structured data — no download
required.**

### Option A — Google Colab (recommended for beginners)
1. Open <https://colab.research.google.com> → *File → Upload notebook*.
2. Upload `notebooks/MH_StepByStep_Guide.ipynb` (or the concise
   `MentalHealth_Crisis_Prediction.ipynb`).
3. *Runtime → Run all.* Figures and a trained model appear under `outputs/`.

### Option B — Locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # macOS XGBoost also needs: brew install libomp
python run.py                          # runs full pipeline, writes outputs/
```

To use **real GLOBEM data**, drop `INS-W_1.csv … INS-W_4.csv` into `data/` and
re-run — the loader auto-detects them, resolves the RAPIDS-style feature columns
by keyword (printing exactly what it matched), and carries weekly survey labels
forward to daily rows. See [`data/README.md`](data/README.md). Validate the
real-data path without the gated download via:
```bash
python tests/test_real_loader.py
```

---

## What it does

Each day, a phone + smartwatch passively record behavioural signals (sleep,
steps, heart-rate variability, night-time screen use, location variety, call
frequency). The model reads the **past two days** of these signals and predicts
whether a crisis will occur in the next 48–72 hours — early enough for a
clinician or family member to intervene.

| Stage | What happens |
|-------|--------------|
| **Features** | 8 raw sensors + 3 engineered (sleep irregularity, activity collapse, social isolation) = 11, each lagged 1 & 2 days → **22 model inputs** |
| **Preprocess** | impute → derive → label → lag → **group-aware split** → Min-Max scale (train-fit only) → SMOTE (train only) |
| **Model** | 5 classifiers compared; XGBoost is the primary model |
| **Explain** | SHAP global + per-prediction feature attribution |
| **Validate** | 5-fold CV + zero-shot cross-dataset test (GLOBEM → StudentLife-like) |

The pipeline trains **only on lagged values** — no same-day data — making it a
genuine 48-hour-ahead *predictor*, not a same-day *detector*.

---

## Example results (synthetic data)

> ⚠️ These numbers come from the **synthetic** generator and are *illustrative
> only*. Replace them with real-GLOBEM results before any paper or presentation.
> The literature-based target table lives in
> [`docs/MentalHealth_Project_Handoff.md`](docs/MentalHealth_Project_Handoff.md).

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|---------:|----------:|-------:|---:|--------:|
| Logistic Regression | 0.89 | 0.72 | 0.87 | 0.78 | **0.947** |
| Random Forest | 0.89 | 0.73 | 0.84 | 0.78 | 0.945 |
| XGBoost | 0.85 | 0.63 | 0.88 | 0.73 | 0.934 |
| SVM | 0.89 | 0.72 | 0.84 | 0.78 | 0.933 |
| Decision Tree | 0.86 | 0.66 | 0.82 | 0.73 | 0.913 |

XGBoost 5-fold CV AUC ≈ 0.95 ± 0.004; cross-dataset AUC drop ≈ 0.03 (≤ 0.08
acceptance threshold). On this synthetic data linear models are competitive —
which is an honest, expected outcome, not a bug. Real behavioural data with
richer non-linear structure is where ensembles typically pull ahead.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run.py                      # CLI: run full pipeline → outputs/
├── src/
│   ├── pipeline.py             # single source of truth (data, preprocess, models, SHAP, figures)
│   └── build_notebooks.py      # regenerates the notebooks from pipeline.py
├── notebooks/
│   ├── MentalHealth_Crisis_Prediction.ipynb   # concise full analysis (30 cells)
│   └── MH_StepByStep_Guide.ipynb              # annotated beginner walkthrough (53 cells)
├── tests/
│   └── test_real_loader.py     # validates the real-GLOBEM path on a mock CSV
├── data/
│   └── README.md               # how to add real GLOBEM / StudentLife CSVs
├── outputs/
│   ├── figures/                # fig1–fig8 (300 DPI)
│   ├── models/                 # xgboost_crisis_model.pkl, minmax_scaler.pkl
│   ├── model_results.csv
│   └── shap_importance.csv
└── docs/
    └── MentalHealth_Project_Handoff.md         # full research handoff (methods, paper guide, Q&A)
```

The notebooks are **self-contained** (they do not import `src/`); the code is
generated from `pipeline.py` via `python src/build_notebooks.py` so the two
never drift.

---

## Figures generated

`fig1_distributions` · `fig2_normalisation` · `fig3_smote` ·
`fig4_correlation` · `fig5_roc_curves` · `fig6_confusion_matrix` ·
`fig7_shap` · `fig8_cross_validation`

---

## Datasets

| | GLOBEM (primary) | StudentLife (validation) |
|--|--|--|
| Source | Univ. of Washington, NeurIPS 2022 | Dartmouth, UbiComp 2014 |
| Use | Training | Zero-shot generalisation test |
| Access | PhysioNet (DUA) | dartmouth.edu |

See [`data/README.md`](data/README.md) for download and setup. Real datasets are
git-ignored (data-use agreements).

---

## Limitations

- Both real datasets are **university students** — results may not generalise to
  clinical, elderly, or working populations.
- PHQ-9 labels are **self-reported and weekly** — they miss intra-week events.
- The model finds **correlation, not causation**.
- Validated on **retrospective** data only; prospective deployment is untested.

---

## Citation / references

GLOBEM: Xu, X. et al. (2022), *NeurIPS 2022*, arXiv:2211.02733 ·
StudentLife: Wang, R. et al. (2014), *ACM UbiComp* ·
SHAP: Lundberg & Lee (2017), *NeurIPS*. Full reference list in the handoff doc.

*Built as an academic research scaffold. Decision-support only — not for clinical use.*
