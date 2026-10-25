"""
Baseline Machine Learning Classifiers
Trains and evaluates Logistic Regression, Random Forest, LightGBM, and XGBoost baseline models.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import lightgbm as lgb

from pathlib import Path
from typing import Dict, Any, List
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
)

from src.utils.config import load_config


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute complete metric suite including Specificity and PR-AUC."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_curve, precision_curve))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": pr_auc,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    return metrics


def train_and_evaluate_baselines(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load extracted features, train baseline models, and save evaluation metrics."""
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    metrics_dir = Path(cfg["paths"]["results_dir"]) / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load k-mer and biophysical features with complete memory ownership
    kmer_data = np.load(processed_dir / "kmer_features.npz")
    biophys_data = np.load(processed_dir / "biophysical_features.npz")

    X_train = np.array(np.hstack([kmer_data["X_train"], biophys_data["X_train"]]), dtype=np.float32, copy=True)
    y_train = np.array(kmer_data["y_train"], dtype=np.float32, copy=True)

    X_val = np.array(np.hstack([kmer_data["X_val"], biophys_data["X_val"]]), dtype=np.float32, copy=True)
    y_val = np.array(kmer_data["y_val"], dtype=np.float32, copy=True)

    X_test = np.array(np.hstack([kmer_data["X_test"], biophys_data["X_test"]]), dtype=np.float32, copy=True)
    y_test = np.array(kmer_data["y_test"], dtype=np.float32, copy=True)

    kmer_data.close()
    biophys_data.close()

    print(f"[Baseline Models] Training feature matrix shape: {X_train.shape}")

    seed = cfg["project"]["seed"]
    
    def get_logistic_regression():
        return LogisticRegression(max_iter=1000, C=1.0, random_state=seed)

    def get_lightgbm():
        return lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=7, random_state=seed, verbose=-1
        )

    def get_random_forest():
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)

    def get_xgboost():
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, random_state=seed, eval_metric="logloss"
        )

    model_factories = [
        ("LogisticRegression", get_logistic_regression),
        ("LightGBM", get_lightgbm),
        ("RandomForest", get_random_forest),
        ("XGBoost", get_xgboost),
    ]

    trained_models = {}
    for name, factory in model_factories:
        print(f"[Baseline Models] Training {name}...")
        clf = factory()
        clf.fit(X_train, y_train)
        trained_models[name] = clf

    results = {}
    summary_rows = []

    # Evaluate all trained models
    for name, clf in trained_models.items():
        print(f"[Baseline Models] Evaluating {name}...")
        # Validation set evaluation
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        val_metrics = compute_comprehensive_metrics(y_val, val_preds, val_probs)

        # Test set evaluation
        test_probs = clf.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        test_metrics = compute_comprehensive_metrics(y_test, test_preds, test_probs)

        results[name] = {"validation": val_metrics, "test": test_metrics}

        # Save model checkpoint
        with open(models_dir / f"baseline_{name.lower()}.pkl", "wb") as f:
            pickle.dump(clf, f)

        summary_rows.append({
            "Model": name,
            "Val_F1": val_metrics["f1_score"],
            "Val_ROC_AUC": val_metrics["roc_auc"],
            "Val_PR_AUC": val_metrics["pr_auc"],
            "Test_F1": test_metrics["f1_score"],
            "Test_ROC_AUC": test_metrics["roc_auc"],
            "Test_PR_AUC": test_metrics["pr_auc"],
            "Test_Recall_Sensitivity": test_metrics["recall_sensitivity"],
            "Test_Specificity": test_metrics["specificity"],
            "Test_Balanced_Acc": test_metrics["balanced_accuracy"],
        })

        print(f"  [{name}] Test F1: {test_metrics['f1_score']:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f} | Test PR-AUC: {test_metrics['pr_auc']:.4f}")

    # Save metrics JSON & CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(metrics_dir / "baseline_summary.csv", index=False)

    with open(metrics_dir / "baseline_details.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Baseline Models] Evaluation complete. Summary saved to: {metrics_dir / 'baseline_summary.csv'}")
    return results


if __name__ == "__main__":
    train_and_evaluate_baselines()
