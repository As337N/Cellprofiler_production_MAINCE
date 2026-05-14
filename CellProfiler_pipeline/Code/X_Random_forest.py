"""
RF_Morphological.py
===================
Random Forest classifier on Cell Painting morphological profiles.
Predicts Metadata_Perturbation from numeric features only.

Usage
-----
python RF_Morphological.py -i <input_csv> -o <output_dir> [-c <cohort>]
                           [--label-col Metadata_Perturbation]
                           [--n-estimators 500]
                           [--ctrl-label DMSO]

Outputs
-------
<output_dir>/
  <cohort>_confusion_matrix.png     — LOO confusion matrix
  <cohort>_per_class_accuracy.png   — per-compound LOO accuracy bar chart
  <cohort>_feature_importances.png  — top-20 feature importances (full fit)
  <cohort>_top_features.csv         — top-30 features with importance scores
  <cohort>_loo_predictions.csv      — per-sample true vs predicted labels
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from typing import Optional


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random Forest on morphological profiles (LOO CV)."
    )
    p.add_argument("-i", "--input",  required=True, type=Path,
                   help="Input path where the is located the CSV with Metadata_ columns + numeric features.")
    p.add_argument("-o", "--output", required=True, type=Path,
                   dest="output_dir", help="Directory for output figures and CSVs.")
    p.add_argument("-c", "--cohort", type=str, default="cohort",
                   help="Cohort name prefix for output files (default: cohort).")
    p.add_argument("--label-col", type=str, default="Metadata_Perturbation",
                   dest="label_col", help="Column to predict (default: Metadata_Perturbation).")
    p.add_argument("--n-estimators", type=int, default=500,
                   dest="n_estimators", help="Number of RF trees (default: 500).")
    p.add_argument("--ctrl-label", type=str, default=None,
                   dest="ctrl_label",
                   help="If set, exclude rows whose label starts with this prefix "
                        "(e.g. DMSO) from classification.")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: Path, label_col: str, ctrl_label: Optional[str]):
    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}.")

    # Optionally exclude control rows
    if ctrl_label:
        before = len(df)
        df = df[~df[label_col].str.startswith(ctrl_label)].reset_index(drop=True)
        print(f"  Excluded {before - len(df)} control rows (prefix='{ctrl_label}')")

    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feat_cols = [
        c for c in df.columns
        if c not in meta_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feat_cols].values
    y = df[label_col].values

    print(f"  Samples  : {X.shape[0]}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Classes  : {len(np.unique(y))}")

    return X, y, feat_cols


# ── Random Forest + LOO ────────────────────────────────────────────────────────

def run_rf_loo(X, y, n_estimators: int, random_state: int = 42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    print(f"\nRunning Leave-One-Out CV (n={len(y)}, {n_estimators} trees)...")
    loo = LeaveOneOut()
    y_pred_enc = cross_val_predict(rf, X, y_enc, cv=loo, n_jobs=-1)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y, y_pred)
    print(f"LOO Accuracy: {acc:.3f}  ({int(acc * len(y))}/{len(y)} correct)")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Full fit for feature importances
    rf.fit(X, y_enc)

    return rf, le, y_pred, acc


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes, acc, save_path: Path):
    n = len(classes)
    figsize = (max(10, n * 0.6), max(8, n * 0.55))
    fig, ax = plt.subplots(figsize=figsize)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(
        f"Random Forest — LOO Confusion Matrix\nAccuracy: {acc:.1%}",
        fontsize=13, pad=14,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_per_class_accuracy(y_true, y_pred, classes, acc, save_path: Path):
    per_class = {
        cls: accuracy_score(y_true[y_true == cls], y_pred[y_true == cls])
        for cls in classes
    }
    s = pd.Series(per_class).sort_values()
    colors = [
        "#e63946" if v < 0.5 else "#457b9d" if v < 1.0 else "#2a9d8f"
        for v in s.values
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.38)))
    ax.barh(s.index, s.values, color=colors, edgecolor="white")
    ax.axvline(acc, color="black", linestyle="--", linewidth=1.2,
               label=f"overall ({acc:.1%})")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("LOO Accuracy", fontsize=11)
    ax.set_title("Per-compound LOO Accuracy", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_feature_importances(rf, feat_cols, save_path: Path, top_n: int = 20):
    importances = pd.Series(rf.feature_importances_, index=feat_cols)
    top = importances.nlargest(top_n)

    colors = ["#2E86AB" if i < top_n // 2 else "#A8DADC" for i in range(top_n)]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index[::-1], top.values[::-1],
            color=colors[::-1], edgecolor="white")
    ax.axvline(top.values.mean(), color="red", linestyle="--",
               linewidth=1, label="mean (top 20)")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title(f"Top-{top_n} Feature Importances (full RF fit)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.cohort

    print(f"\n{'─'*55}")
    path_input_csv = args.input / f"{args.cohort}_normalized.csv"
    print(f"Loading: {path_input_csv}")
    X, y, feat_cols = load_data(path_input_csv, args.label_col, args.ctrl_label)

    rf, le, y_pred, acc = run_rf_loo(X, y, args.n_estimators)
    classes = le.classes_

    print(f"\nSaving outputs to {args.output_dir}/")

    plot_confusion_matrix(
        y, y_pred, classes, acc,
        save_path=Path(f"{prefix}_confusion_matrix.png"),
    )
    plot_per_class_accuracy(
        y, y_pred, classes, acc,
        save_path=Path(f"{prefix}_per_class_accuracy.png"),
    )
    plot_feature_importances(
        rf, feat_cols,
        save_path=Path(f"{prefix}_feature_importances.png"),
    )

    # CSVs
    importances = pd.Series(rf.feature_importances_, index=feat_cols)
    importances.nlargest(30).reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    ).to_csv(f"{prefix}_top_features.csv", index=False)
    print(f"  Saved → {prefix}_top_features.csv")

    pd.DataFrame({"true": y, "predicted": y_pred}).to_csv(
        f"{prefix}_loo_predictions.csv", index=False
    )
    print(f"  Saved → {prefix}_loo_predictions.csv")

    print(f"\n{'─'*55}")
    print(f"Done. LOO Accuracy: {acc:.1%}")


if __name__ == "__main__":
    main()