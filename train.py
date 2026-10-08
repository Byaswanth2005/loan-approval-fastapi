"""Train the loan approval model with Pandas + scikit-learn and record metrics.

Usage:
    python train.py                      # uses data/loan_data.csv (real dataset)
    python train.py --synthetic          # demo data ONLY for testing the pipeline
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/loan_data.csv")
SAMPLE_PATH = Path("data/sample_loan_data.csv")
MODEL_PATH = Path("model/model.joblib")
METRICS_PATH = Path("model/metrics.json")

CATEGORICAL = ["Gender", "Married", "Dependents", "Education",
               "Self_Employed", "Property_Area"]
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
           "Loan_Amount_Term", "Credit_History"]
TARGET = "Loan_Status"  # Y / N


def make_synthetic(n=600, seed=42):
    """Fake data so you can test the code. Do NOT quote metrics from this."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], n),
        "Married": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["0", "1", "2", "3+"], n),
        "Education": rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22]),
        "Self_Employed": rng.choice(["Yes", "No"], n, p=[0.14, 0.86]),
        "ApplicantIncome": rng.integers(1500, 20000, n),
        "CoapplicantIncome": rng.integers(0, 8000, n),
        "LoanAmount": rng.integers(50, 500, n).astype(float),
        "Loan_Amount_Term": rng.choice([360, 180, 120], n, p=[0.85, 0.1, 0.05]).astype(float),
        "Credit_History": rng.choice([1.0, 0.0], n, p=[0.84, 0.16]),
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n),
    })
    score = 2.5 * df["Credit_History"] + df["ApplicantIncome"] / 20000 - df["LoanAmount"] / 500
    prob = 1 / (1 + np.exp(-(score - 1.2) * 2))
    df[TARGET] = np.where(rng.random(n) < prob, "Y", "N")
    return df


def load_data(synthetic: bool):
    """Returns (dataframe, label). Prefers the real dataset over the sample."""
    if synthetic:
        print("WARNING: using SYNTHETIC data. Metrics are not real.")
        return make_synthetic(), "synthetic (demo only)"
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH), str(DATA_PATH)
    if SAMPLE_PATH.exists():
        print("=" * 60)
        print("WARNING: data/loan_data.csv not found. Using the bundled SAMPLE")
        print("(synthetic) data so the project runs. Do NOT put these metrics")
        print("on your resume. Add the real Kaggle file as data/loan_data.csv")
        print("and run train.py again.")
        print("=" * 60)
        return pd.read_csv(SAMPLE_PATH), "SAMPLE synthetic data (replace with real)"
    raise SystemExit(
        f"Missing {DATA_PATH}. Download the Kaggle 'Loan Prediction' dataset "
        "(train.csv), save it as data/loan_data.csv, then rerun."
    )


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def feature_importance(pipe, top=8):
    """Top features as percentages (coefficient size or tree importance)."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    model = pipe.named_steps["model"]
    raw = np.abs(model.coef_[0]) if hasattr(model, "coef_") else model.feature_importances_
    pct = raw / raw.sum() * 100
    pairs = sorted(zip(names, pct), key=lambda t: -t[1])[:top]
    return [{"feature": n.split("__", 1)[-1].replace("_", " "),
             "importance": round(float(v), 2)} for n, v in pairs]


def main(synthetic: bool):
    df, label = load_data(synthetic)
    df = df.dropna(subset=[TARGET])
    df["Dependents"] = df["Dependents"].astype("string")
    X = df[CATEGORICAL + NUMERIC]
    y = (df[TARGET] == "Y").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_cv, results = None, -1.0, {}
    for name, model in candidates.items():
        scores = cross_val_score(build_pipeline(model), X_train, y_train,
                                 cv=cv, scoring="accuracy")
        results[name] = round(float(scores.mean()), 4)
        print(f"{name}: CV accuracy = {scores.mean():.4f}")
        if scores.mean() > best_cv:
            best_name, best_cv = name, scores.mean()

    pipe = build_pipeline(candidates[best_name]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "dataset": label,
        "best_model": best_name,
        "cv_accuracy": results,
        "test_rows": int(len(y_test)),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    metrics["feature_importance"] = feature_importance(pipe)
    print(json.dumps(metrics, indent=2))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
