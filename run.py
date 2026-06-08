#!/usr/bin/env python3
"""Command-line entry point: run the full pipeline and save all outputs.

    python run.py

Uses real GLOBEM CSVs from ./data if present (INS-W_1.csv … INS-W_4.csv),
otherwise synthetic GLOBEM-structured data. Figures and artifacts are written
under ./outputs. Paths are resolved relative to this file, so it works from any
working directory.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pipeline as p  # noqa: E402


def main():
    res, shap_df, (ag, asl), is_real = p.run_all(
        data_dir=os.path.join(ROOT, "data"),
        out_dir=os.path.join(ROOT, "outputs"),
    )
    print("\n=== Model results ===")
    print(res.to_string(index=False))
    print(f"\nCross-dataset AUC: GLOBEM={ag:.3f}  StudentLife={asl:.3f}  drop={ag-asl:.3f}")
    print(f"Data source: {'REAL GLOBEM' if is_real else 'SYNTHETIC'}")
    print(f"\nTop SHAP features:\n{shap_df.head(5).to_string(index=False)}")
    print("\nFigures + artifacts saved under outputs/.")


if __name__ == "__main__":
    main()
