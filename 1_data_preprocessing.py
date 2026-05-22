"""
1. Data Preprocessing
=====================
Loads the raw CSV, renames psychological variables, recodes sex,
ensures binary variables are numeric 0/1, and creates derived columns
(log-transformed symptom duration, GSI fallback).

Usage
-----
    python 1_data_preprocessing.py

Output
------
    outputs/preprocessed_data.csv
"""

from config import OUTDIR, load_and_prepare_data


if __name__ == "__main__":
    df = load_and_prepare_data()
    out_path = OUTDIR / "preprocessed_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Preprocessed data saved → {out_path}  (shape: {df.shape})")
