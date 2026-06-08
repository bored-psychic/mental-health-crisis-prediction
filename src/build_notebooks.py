"""Generate the two project notebooks from the tested pipeline module.

Both notebooks are self-contained (no import of src/pipeline) so they run on a
fresh Google Colab runtime. The function source is pulled directly from
pipeline.py via inspect, guaranteeing the notebook code matches the module that
was tested locally.

    python src/build_notebooks.py
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pipeline as p  # noqa: E402
import nbformat as nbf  # noqa: E402
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(HERE, "notebooks")


def src(*fns):
    """Concatenated source of one or more functions, stripped of trailing ws."""
    return "\n\n".join(inspect.getsource(f).rstrip() for f in fns)


# --------------------------------------------------------------------------- #
# Shared cell fragments
# --------------------------------------------------------------------------- #

INSTALL = (
    "# Colab already has most of these; this is a no-op there. Run once locally.\n"
    "# !pip install -q numpy pandas scikit-learn matplotlib seaborn xgboost shap imbalanced-learn joblib\n"
    "print('Dependencies ready.')"
)

IMPORTS = (
    "import os\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "warnings.filterwarnings('ignore')\n"
    "pd.set_option('display.width', 120)\n"
    "RANDOM_STATE = 42\n"
    "print('numpy', np.__version__, '| pandas', pd.__version__)"
)

FEATURE_DEFS = src(p.lag_names) + "\n\n" + (
    'RAW_FEATURES = [\n'
    '    "sleep_hours", "sleep_onset", "step_count", "hrv_ms",\n'
    '    "screen_total", "screen_night", "loc_entropy", "call_freq",\n'
    ']\n'
    'DERIVED_FEATURES = ["sleep_irregularity", "activity_collapse", "social_isolation"]\n'
    'ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES            # 11 features\n'
    'COMMON_FEATURES = ["sleep_hours", "sleep_onset", "step_count",\n'
    '                   "loc_entropy", "call_freq"]            # shared w/ StudentLife\n'
    'PHQ9_CRISIS_THRESHOLD = 10\n'
    'ALL_LAG_FEATURES = lag_names(ALL_FEATURES)                # 22 model inputs\n'
    'COMMON_LAG_FEATURES = lag_names(COMMON_FEATURES)          # 10 cross-dataset inputs\n'
    'print(f"{len(ALL_LAG_FEATURES)} lag features feed the model")'
)

def _consts(*names):
    """Embed module-level constants as assignable source."""
    import pprint
    out = []
    for n in names:
        out.append(f"{n} = " + pprint.pformat(getattr(p, n), width=88, sort_dicts=False))
    return "\n".join(out)

DATA_DEFS = (
    src(p.make_synthetic_globem)
    + "\n\n" + _consts("FEATURE_PATTERNS", "FEATURE_EXCLUDE", "LABEL_KEYWORDS",
                       "PID_KEYWORDS", "DATE_KEYWORDS")
    + "\n\n" + src(p._resolve_column, p._first_present, p._map_real_globem, p.load_globem)
)

PREPROC_DEFS = src(p.impute, p.derive_features, p.add_label, p.make_lags, p.build_dataset)
SPLIT_DEFS = src(p.split_normalise_balance)
MODEL_DEFS = src(p.build_models, p.evaluate, p.train_all, p.cross_validate_auc)
CROSS_DEFS = src(p.make_synthetic_studentlife, p.impute_common,
                 p.derive_common, p.cross_dataset_validation)
SHAP_DEFS = src(p.shap_importance)
FIG_DEFS = src(p._fig_dir, p.generate_figures, p.save_artifacts)

SHOW_FIGS = (
    "from IPython.display import Image, display\n"
    "for f in sorted(os.listdir('outputs/figures')):\n"
    "    if f.endswith('.png'):\n"
    "        print(f)\n"
    "        display(Image(filename=os.path.join('outputs/figures', f)))"
)


def title_cell():
    return new_markdown_cell(
        "# 🧠 Mental Health Crisis Prediction\n"
        "### Predicting a PHQ-9 crisis (score ≥ 10) 48–72 hours in advance "
        "from passive sensing data\n\n"
        "Five ML models (Logistic Regression, Decision Tree, SVM, Random Forest, "
        "**XGBoost**) are compared on 22 time-lagged behavioural features, with "
        "SHAP explainability and cross-dataset validation.\n\n"
        "> **Data:** runs on synthetic GLOBEM-structured data out of the box. "
        "Drop `INS-W_1.csv … INS-W_4.csv` next to this notebook to use real "
        "GLOBEM data (the loader auto-detects them).\n\n"
        "> ⚠️ **Decision-support only — not a diagnostic tool.** Metric values "
        "on synthetic data are illustrative, not the literature targets in the "
        "handoff."
    )


# --------------------------------------------------------------------------- #
# Full analysis notebook
# --------------------------------------------------------------------------- #

def build_full():
    cells = [
        title_cell(),
        new_markdown_cell("## Setup\nInstall dependencies (no-op on Colab) and import."),
        new_code_cell(INSTALL),
        new_code_cell(IMPORTS),

        new_markdown_cell(
            "## Feature definitions\n"
            "8 raw sensor features + 3 engineered features = **11 features**, each "
            "lagged by 1 and 2 days → **22 model inputs**. Training on *only* lagged "
            "values is what makes this a 48-hour-ahead predictor, not same-day detection."),
        new_code_cell(FEATURE_DEFS),

        new_markdown_cell(
            "## 1. Load data\n"
            "Synthetic generator models a latent daily *distress* level that drives "
            "every sensor channel, plus a non-linear compounding term (short sleep "
            "**and** high night-screen **and** low activity together spike risk). "
            "PHQ-9 is derived from this, so a genuine — non-trivial — signal exists."),
        new_code_cell(DATA_DEFS),
        new_code_cell(
            "df, is_real = load_globem('.')\n"
            "print('source:', 'REAL GLOBEM' if is_real else 'SYNTHETIC',\n"
            "      '| user-days:', len(df), '| users:', df['user_id'].nunique())\n"
            "df.head()"),

        new_markdown_cell(
            "## 2. Preprocessing\n"
            "`impute → derive → label → lag`. Imputation: forward-fill short gaps, "
            "then personal mean, then global mean. Derived features add sleep "
            "irregularity, activity collapse, and a social-isolation index."),
        new_code_cell(PREPROC_DEFS),
        new_code_cell(
            "ds = build_dataset(df)\n"
            "print('rows after lagging:', len(ds),\n"
            "      '| crisis rate:', round(ds['crisis_label'].mean(), 3))\n"
            "ds[['user_id','day','crisis_label'] + ALL_LAG_FEATURES[:3]].head()"),

        new_markdown_cell(
            "## 3. Split → normalise → SMOTE\n"
            "**Group-aware** split (no user appears in both train and test — avoids "
            "leakage). The scaler is fit on **train only**; SMOTE balances the "
            "**training set only**. Test data is never resampled or peeked at."),
        new_code_cell(SPLIT_DEFS),
        new_code_cell(
            "data = split_normalise_balance(ds)\n"
            "print('train', data['X_train'].shape, '| test', data['X_test'].shape,\n"
            "      '| balanced train', data['X_train_bal'].shape)\n"
            "print('test crisis rate (untouched):', round(data['y_test'].mean(), 3))"),

        new_markdown_cell(
            "## 4. Train & compare five models\n"
            "Hyper-parameters follow the handoff. AUC-ROC is the primary metric "
            "because accuracy is misleading under class imbalance."),
        new_code_cell(MODEL_DEFS),
        new_code_cell(
            "results_df, fitted, probas = train_all(data)\n"
            "results_df.round(3)"),
        new_code_cell(
            "cv = cross_validate_auc(ds)\n"
            "print(f'XGBoost 5-fold CV AUC: {cv.mean():.3f} ± {cv.std():.3f}')"),

        new_markdown_cell(
            "## 5. Cross-dataset validation\n"
            "Train a reduced XGBoost on GLOBEM's 5 common features, then apply it "
            "**zero-shot** to a phone-only StudentLife-like cohort. A drop ≤ 0.08 "
            "AUC indicates the model learned generalisable behavioural signals."),
        new_code_cell(CROSS_DEFS),
        new_code_cell(
            "auc_globem, auc_sl = cross_dataset_validation(df)\n"
            "print(f'GLOBEM AUC {auc_globem:.3f} -> StudentLife AUC {auc_sl:.3f} '\n"
            "      f'(drop {auc_globem-auc_sl:.3f})')"),

        new_markdown_cell(
            "## 6. SHAP explainability\n"
            "Mean |SHAP| ranks which lagged behavioural signals drive crisis "
            "predictions — the clinical interpretability that black-box models lack."),
        new_code_cell(SHAP_DEFS),
        new_code_cell(
            "shap_df, shap_values = shap_importance(\n"
            "    fitted['XGBoost'], data['X_test'], data['feature_cols'])\n"
            "shap_df.head(10)"),

        new_markdown_cell(
            "## 7. Figures & saved artifacts\n"
            "Generates the 8 project figures (300 DPI) and saves the trained model, "
            "scaler, and result CSVs under `outputs/`."),
        new_code_cell(FIG_DEFS),
        new_code_cell(
            "paths = generate_figures(ds, data, results_df, fitted, probas,\n"
            "                         shap_df, shap_values, auc_globem, auc_sl)\n"
            "save_artifacts(fitted, data['scaler'], results_df, shap_df)\n"
            "print('saved', len(paths), 'figures + model/scaler/CSVs under outputs/')"),
        new_code_cell(SHOW_FIGS),

        new_markdown_cell(
            "## Notes\n"
            "- **Synthetic results are illustrative.** Replace with real-GLOBEM "
            "numbers before any paper/presentation; literature targets live in the "
            "handoff.\n"
            "- Model ranking can differ from the handoff's expected table — on "
            "synthetic data linear models are competitive. A null finding is still "
            "a valid result.\n"
            "- This is a **decision-support** prototype, not a diagnostic device."),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    return nb


# --------------------------------------------------------------------------- #
# Beginner step-by-step notebook
# --------------------------------------------------------------------------- #

def tip(text):
    return new_markdown_cell(f"> 💡 **What just happened:** {text}")


def build_stepbystep():
    cells = [
        new_markdown_cell(
            "# 🧠 Mental Health Crisis Prediction — Step-by-Step Guide\n"
            "A beginner-friendly walkthrough. **Run each cell with Shift+Enter**, "
            "read the output, then continue. Every step has a 💡 note explaining "
            "what happened.\n\n"
            "By the end you will have trained five models, explained them with "
            "SHAP, validated across datasets, and saved all figures — using "
            "synthetic data that mimics the real GLOBEM dataset."),

        new_markdown_cell(
            "## Step 0 — Install & import\n"
            "Colab already has these libraries. Locally, uncomment the pip line."),
        new_code_cell(INSTALL),
        new_code_cell(IMPORTS),
        tip("We loaded NumPy/Pandas and fixed a random seed (42) so every run is "
            "reproducible — you and your supervisor get identical numbers."),

        new_markdown_cell(
            "## Step 1 — What are we predicting?\n"
            "**Target:** will this person's PHQ-9 depression score cross 10 in the "
            "next 48–72 hours? This is **binary classification**: 1 = crisis, "
            "0 = no crisis. We use *yesterday's* and *the day before's* behaviour "
            "to predict *tomorrow* — a genuine early-warning system."),

        new_markdown_cell("## Step 2 — Define the features"),
        new_code_cell(FEATURE_DEFS),
        tip("8 phone/wearable signals + 3 engineered signals = 11 features. Each is "
            "copied to 'yesterday' (lag1) and 'two days ago' (lag2), giving the "
            "22 numbers the model actually sees."),

        new_markdown_cell(
            "## Step 3 — Get the data\n"
            "If real GLOBEM CSVs (`INS-W_1.csv`…`INS-W_4.csv`) are present we use "
            "them; otherwise we generate realistic synthetic data with the same "
            "structure."),
        new_code_cell(DATA_DEFS),
        new_code_cell(
            "df, is_real = load_globem('.')\n"
            "print('Using', 'REAL' if is_real else 'SYNTHETIC', 'data')\n"
            "print(df.shape[0], 'user-days from', df['user_id'].nunique(), 'people')\n"
            "df.head()"),
        tip("Each row = one person on one day. Columns are the raw sensor readings "
            "plus that week's PHQ-9 score. Notice some values are NaN (missing) — "
            "real sensors drop out, so we simulate that too."),

        new_markdown_cell(
            "## Step 4 — Fill in missing values (imputation)\n"
            "Strategy: copy the last reading for short gaps, else use the person's "
            "own average, else the global average."),
        new_code_cell(src(p.impute)),
        new_code_cell(
            "before = df[RAW_FEATURES].isna().sum().sum()\n"
            "df = impute(df)\n"
            "after = df[RAW_FEATURES].isna().sum().sum()\n"
            "print(f'Missing values: {before} -> {after}')"),
        tip("We filled every gap. Using each person's *own* average is more honest "
            "than a population average — night owls stay night owls."),

        new_markdown_cell(
            "## Step 5 — Engineer 3 extra features\n"
            "Raw sensors aren't enough. We compute **sleep irregularity** (how "
            "chaotic bedtimes are), **activity collapse** (% step drop vs personal "
            "baseline), and a **social isolation** index."),
        new_code_cell(src(p.derive_features)),
        new_code_cell(
            "df = derive_features(df)\n"
            "df[['user_id','day'] + DERIVED_FEATURES].head()"),
        tip("These capture *change relative to a person's own normal* — a 50% step "
            "drop matters whether you usually walk 4,000 or 12,000 steps."),

        new_markdown_cell(
            "## Step 6 — Create the crisis label\n"
            "PHQ-9 ≥ 10 becomes label 1 (crisis), below 10 becomes 0."),
        new_code_cell(src(p.add_label)),
        new_code_cell(
            "df = add_label(df)\n"
            "print(df['crisis_label'].value_counts())\n"
            "print('crisis rate:', round(df['crisis_label'].mean(), 3))"),
        tip("Far more no-crisis days than crisis days — this imbalance is why we "
            "later use SMOTE and judge models by AUC-ROC, not accuracy."),

        new_markdown_cell(
            "## Step 7 — Make the 48-hour lag features\n"
            "Shift each feature back 1 and 2 days, then drop the first 2 days per "
            "person (no history yet)."),
        new_code_cell(src(p.make_lags)),
        new_code_cell(
            "ds = make_lags(df, features=ALL_FEATURES)\n"
            "print('rows with full 2-day history:', len(ds))\n"
            "ds[['user_id','day','crisis_label','sleep_hours_lag1','sleep_hours_lag2']].head()"),
        tip("The model now only ever sees PAST data to predict the FUTURE label — "
            "no cheating with same-day values."),

        new_markdown_cell(
            "## Step 8 — Split, scale, and balance\n"
            "Split by *person* (group-aware) so no one is in both train and test. "
            "Scale to [0,1] (fit on train only). Balance the training set with "
            "SMOTE."),
        new_code_cell(SPLIT_DEFS),
        new_code_cell(
            "data = split_normalise_balance(ds)\n"
            "print('train rows:', len(data['y_train']),\n"
            "      '| after SMOTE:', len(data['y_train_bal']),\n"
            "      '| test rows:', len(data['y_test']))"),
        tip("SMOTE invents *synthetic* crisis examples (not copies) so the model "
            "sees a 50/50 balance and stops ignoring the rare crisis class. The "
            "test set is left untouched so our score is honest."),

        new_markdown_cell(
            "## Step 9 — Train all five models\n"
            "From a simple Logistic Regression baseline up to XGBoost."),
        new_code_cell(MODEL_DEFS),
        new_code_cell(
            "results_df, fitted, probas = train_all(data)\n"
            "results_df.round(3)"),
        tip("Read the **AUC-ROC** column — it measures how well each model ranks a "
            "real crisis above a non-crisis day, regardless of threshold. 0.50 = "
            "coin flip, 1.0 = perfect."),
        new_code_cell(
            "cv = cross_validate_auc(ds)\n"
            "print(f'XGBoost cross-validated AUC: {cv.mean():.3f} ± {cv.std():.3f}')"),
        tip("5-fold cross-validation confirms the score is stable, not a lucky "
            "single split."),

        new_markdown_cell(
            "## Step 10 — Does it generalise? (cross-dataset)\n"
            "Train on GLOBEM's shared features, test zero-shot on a phone-only "
            "StudentLife-like cohort."),
        # build_dataset wraps the Step 4-7 functions; cross-validation needs it.
        new_code_cell(src(p.build_dataset) + "\n\n" + CROSS_DEFS),
        new_code_cell(
            "auc_globem, auc_sl = cross_dataset_validation(df)\n"
            "print(f'GLOBEM {auc_globem:.3f} -> StudentLife {auc_sl:.3f} '\n"
            "      f'(drop {auc_globem - auc_sl:.3f}; ≤0.08 is good)')"),
        tip("A small drop means the model learned *universal* behavioural signs of "
            "deterioration, not patterns unique to one study."),

        new_markdown_cell(
            "## Step 11 — Explain the model with SHAP\n"
            "Which signals drove the predictions?"),
        new_code_cell(SHAP_DEFS),
        new_code_cell(
            "shap_df, shap_values = shap_importance(\n"
            "    fitted['XGBoost'], data['X_test'], data['feature_cols'])\n"
            "shap_df.head(10)"),
        tip("A clinician can now see *why* an alert fired (e.g. 'sleep dropped and "
            "night phone use rose two days ago') instead of trusting a black box."),

        new_markdown_cell(
            "## Step 12 — Save figures and the model\n"
            "Creates all 8 figures (300 DPI) plus the trained model, scaler and "
            "CSV results under `outputs/`."),
        new_code_cell(FIG_DEFS),
        new_code_cell(
            "paths = generate_figures(ds, data, results_df, fitted, probas,\n"
            "                         shap_df, shap_values, auc_globem, auc_sl)\n"
            "save_artifacts(fitted, data['scaler'], results_df, shap_df)\n"
            "print('Saved', len(paths), 'figures and all artifacts.')"),
        new_code_cell(SHOW_FIGS),
        tip("Everything you need for the paper/presentation is now in `outputs/`. "
            "Remember: these numbers are from synthetic data — rerun with real "
            "GLOBEM CSVs before reporting final results."),

        new_markdown_cell(
            "## 🎓 You're done!\n"
            "You built an end-to-end, explainable, cross-validated crisis-prediction "
            "pipeline. Next: register on PhysioNet, download the real GLOBEM CSVs, "
            "drop them next to this notebook, and re-run every cell to get real "
            "results.\n\n"
            "*Decision-support prototype only — not for clinical diagnosis.*"),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    return nb


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    full = build_full()
    step = build_stepbystep()
    with open(os.path.join(NB_DIR, "MentalHealth_Crisis_Prediction.ipynb"), "w") as f:
        nbf.write(full, f)
    with open(os.path.join(NB_DIR, "MH_StepByStep_Guide.ipynb"), "w") as f:
        nbf.write(step, f)
    print(f"Full notebook:  {len(full.cells)} cells")
    print(f"Step notebook:  {len(step.cells)} cells")


if __name__ == "__main__":
    main()
