"""Verify the real-GLOBEM data path without needing the real (gated) download.

We fabricate a small CSV that uses *authentic GLOBEM-style* column names
(RAPIDS feature naming + weekly survey labels), write it as INS-W_1.csv, then
push it through the real loader and the full modelling pipeline. This proves the
keyword resolver, weekly->daily label carry-forward, and downstream training all
work on realistically-named data.

    python tests/test_real_loader.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pipeline as p  # noqa: E402


# Authentic-looking GLOBEM/RAPIDS column names for each of our 8 features.
GLOBEM_NAMES = {
    "sleep_hours": "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:allday",
    "sleep_onset": "f_slp:fitbit_sleep_intraday_rapids_starttimefirstmainsleep:allday",
    "step_count": "f_steps:fitbit_steps_intraday_rapids_sumsteps:allday",
    "hrv_ms": "f_steps:fitbit_heartrate_summary_rapids_restinghr:allday",
    "screen_total": "f_screen:phone_screen_rapids_sumdurationunlock:allday",
    "screen_night": "f_screen:phone_screen_rapids_sumdurationunlock:night",
    "loc_entropy": "f_loc:phone_locations_doryab_locationentropy:allday",
    "call_freq": "f_call:phone_calls_rapids_all_count:allday",
}
LABEL_NAME = "phq9_total"  # weekly survey score


def make_mock_globem_csv(path, n_users=30, days=84, seed=1):
    """Synthetic data relabelled with GLOBEM-style headers; labels weekly-sparse."""
    df = p.make_synthetic_globem(n_users=n_users, days=days, seed=seed)
    real = pd.DataFrame()
    real["pid"] = "INS-W-" + df["user_id"].astype(str)
    real["date"] = pd.Timestamp("2019-01-07") + pd.to_timedelta(df["day"], unit="D")
    for feat, col in GLOBEM_NAMES.items():
        real[col] = df[feat]
    # PHQ-9 known only on day 0, 7, 14, ... (weekly survey); NaN otherwise.
    weekly = df["phq9"].where(df["day"] % 7 == 0, np.nan)
    real[LABEL_NAME] = weekly.values
    # Sprinkle extra distractor columns the resolver must ignore.
    real["f_steps:fitbit_steps_intraday_rapids_maxsteps:hour"] = df["step_count"] / 10
    real["dataset_id"] = 1
    real.to_csv(path, index=False)
    return real


def main():
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "INS-W_1.csv")
        mock = make_mock_globem_csv(csv)
        weekly_labelled = mock[LABEL_NAME].notna().sum()
        print(f"Mock CSV: {mock.shape[0]} rows, {mock.shape[1]} cols, "
              f"{weekly_labelled} weekly-labelled days.")

        df, is_real = p.load_globem(tmp, verbose=True)
        assert is_real, "loader did not detect the real CSV path"

        # Every raw feature must have resolved (no all-NaN columns).
        for f in p.RAW_FEATURES:
            assert df[f].notna().any(), f"feature {f} failed to resolve"
        # Carry-forward must densify weekly labels to (almost) every day.
        assert len(df) > weekly_labelled * 3, "weekly->daily carry-forward did not expand labels"
        assert df["phq9"].notna().all(), "phq9 should be dense after carry-forward"

        # Full pipeline must run and train on the real-path frame.
        ds = p.build_dataset(df)
        data = p.split_normalise_balance(ds)
        results, fitted, _ = p.train_all(data)
        auc = results["AUC-ROC"].max()
        print(f"\nReal-path pipeline OK -> {len(ds)} modelling rows, "
              f"best AUC {auc:.3f}")
        assert 0.5 <= auc <= 1.0
        assert "screen_night" in df.columns and df["screen_night"].notna().any(), \
            "night-screen column should resolve to the :night variant"

        print("\nALL REAL-PATH CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
