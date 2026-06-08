# CLAUDE.md — project guide for Claude

This file orients Claude (and anyone new) to this repository. If you are Claude
helping the user, read this first, then help them run, understand, or extend the
project. Be friendly and concrete — assume the user may be non-technical.

## What this project is

An end-to-end machine-learning pipeline that **predicts a mental-health crisis
(PHQ-9 depression score ≥ 10) 48–72 hours in advance** from passive smartphone
and wearable sensing data. It compares five models (Logistic Regression,
Decision Tree, SVM, Random Forest, **XGBoost**), explains them with SHAP, and
tests whether the model generalises across two datasets.

> ⚠️ **Decision-support prototype only — NOT a diagnostic or treatment tool.**
> It is an academic / educational project. Any real use would require clinician
> review and ethics approval.

## How to run it (two easy paths)

**Easiest — Google Colab (no setup):**
1. Open <https://colab.research.google.com> → *File → Upload notebook*.
2. Upload `notebooks/MH_StepByStep_Guide.ipynb` (beginner, fully explained) or
   `notebooks/MentalHealth_Crisis_Prediction.ipynb` (concise).
3. *Runtime → Run all.* It generates results and figures automatically.

**Locally (on this computer):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # macOS XGBoost also needs: brew install libomp
python run.py                          # runs everything, writes outputs/
```

The project **runs on synthetic (made-up but realistic) data by default**, so it
works with zero downloads. To use the real GLOBEM dataset, place
`INS-W_1.csv … INS-W_4.csv` in `data/` and re-run — see `data/README.md`.

## Where things live

| Path | What it is |
|------|------------|
| `src/pipeline.py` | The brain of the project — all the data, preprocessing, models, SHAP, and figure code. Single source of truth. |
| `src/build_notebooks.py` | Regenerates the two notebooks from `pipeline.py` (keeps them in sync). |
| `notebooks/` | The two runnable notebooks (concise + beginner walkthrough). |
| `run.py` | One command to run the whole pipeline from the terminal. |
| `tests/test_real_loader.py` | Checks the real-dataset loading path works. |
| `outputs/figures/` | The 8 generated charts (fig1–fig8). |
| `outputs/models/` | The trained model + scaler (created when you run it). |
| `docs/MentalHealth_Project_Handoff.md` | The full research write-up: methods, paper-writing guide, journal targets, Q&A prep. |
| `README.md` | Project overview and results. |

## Important things to know (please honour these)

- **Synthetic results are illustrative, not real.** The example numbers (AUC ≈
  0.91–0.95) come from made-up data. They must be replaced with real-GLOBEM
  results before any paper or presentation. The literature-based *target*
  numbers live in `docs/MentalHealth_Project_Handoff.md`.
- **On synthetic data, XGBoost does not always "win."** Logistic Regression and
  Random Forest are competitive. That is honest and expected — do not tweak the
  data just to make XGBoost look best.
- The notebooks are **generated** from `src/pipeline.py`. If you change the
  analysis code, edit `pipeline.py` and then run
  `python src/build_notebooks.py` to rebuild the notebooks — don't hand-edit the
  notebook code cells, or the two will drift apart.
- `.venv/` and real dataset CSVs are intentionally not committed to git.

## Good things to ask Claude for

- "Run the project and show me the results."
- "Explain what figure 7 (SHAP) means in plain English."
- "I downloaded the real GLOBEM files — help me plug them in."
- "Help me write the Methods section of the paper from this code."
- "Change the crisis threshold / add a new feature and re-run."
- "Walk me through the step-by-step notebook one cell at a time."

## Verify before claiming success

After running, check that `outputs/figures/` has `fig1…fig8` and that
`python run.py` printed a model results table without errors. The real-data path
can be sanity-checked any time with `python tests/test_real_loader.py` (needs no
download).
