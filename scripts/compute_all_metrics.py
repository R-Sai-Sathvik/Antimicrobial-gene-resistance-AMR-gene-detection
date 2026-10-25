import sys
import os
import json
import pickle
import numpy as np
import pandas as pd

# Crucial Windows fix: Import LightGBM first
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, precision_recall_curve,
    auc, confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import load_config
from src.models.baseline_models import compute_comprehensive_metrics

def main():
    print("=" * 80)
    print(" COMPUTING COMPREHENSIVE BENCHMARK METRICS FOR ALL MODELS & ENSEMBLE")
    print("=" * 80)

    cfg = load_config("configs/config.yaml")
    processed_dir = "data/processed"
    models_dir = "models"
    results_dir = "results/metrics"

    # 1. Load test features and labels
    print("[1/5] Loading test features and ground-truth labels...")
    kmer_npz = np.load(os.path.join(processed_dir, "kmer_features.npz"))
    bio_npz = np.load(os.path.join(processed_dir, "biophysical_features.npz"))
    
    X_test = np.ascontiguousarray(np.hstack([kmer_npz["X_test"], bio_npz["X_test"]]), dtype=np.float32)
    y_test = np.ascontiguousarray(kmer_npz["y_test"], dtype=np.int32)
    test_df = pd.read_csv(os.path.join(processed_dir, "test_split.csv"))
    print(f"      Total test sequences: {len(y_test)} (Positives: {sum(y_test==1)}, Negatives: {sum(y_test==0)})")

    # 2. Evaluate all 4 Baseline ML Models
    print("\n[2/5] Evaluating all 4 Baseline ML models...")
    baseline_files = {
        "LogisticRegression": "baseline_logisticregression.pkl",
        "RandomForest": "baseline_randomforest.pkl",
        "XGBoost": "baseline_xgboost.pkl",
        "LightGBM": "lightgbm_model.pkl"
    }

    test_probs = {}
    model_metrics = {}

    for model_name, pkl_file in baseline_files.items():
        pkl_path = os.path.join(models_dir, pkl_file)
        if not os.path.exists(pkl_path):
            # check alternate name
            alt_path = os.path.join(models_dir, f"baseline_{model_name.lower()}.pkl")
            if os.path.exists(alt_path):
                pkl_path = alt_path
        
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)
        test_probs[model_name] = prob
        model_metrics[model_name] = compute_comprehensive_metrics(y_test, pred, prob)
        print(f"      {model_name:20s} -> F1: {model_metrics[model_name]['f1_score']:.4f} | Acc: {model_metrics[model_name]['accuracy']:.4f} | ROC-AUC: {model_metrics[model_name]['roc_auc']:.4f}")

    # 3. Evaluate ESM-2 35M Transformer
    print("\n[3/5] Evaluating fine-tuned ESM-2 35M Protein Transformer...")
    import torch
    from transformers import AutoTokenizer
    from src.models.esm2_model import ESM2Classifier, ProteinSequenceDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Inference device: {device}")

    esm_name = cfg["features"]["esm"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(esm_name)
    esm_model = ESM2Classifier(model_name=esm_name).to(device)

    esm_ckpt = os.path.join(models_dir, "esm2_amr_classifier.pt")
    esm_model.load_state_dict(torch.load(esm_ckpt, map_location=device, weights_only=False))
    esm_model.eval()

    test_ds = ProteinSequenceDataset(test_df["sequence"].tolist(), test_df["label"].tolist(), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    esm_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, _ = esm_model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            esm_probs.extend(probs)

    esm_probs = np.array(esm_probs)
    esm_pred = (esm_probs >= 0.50).astype(int)
    test_probs["ESM2"] = esm_probs
    model_metrics["ESM2"] = compute_comprehensive_metrics(y_test, esm_pred, esm_probs)
    print(f"      {'ESM-2 (35M)':20s} -> F1: {model_metrics['ESM2']['f1_score']:.4f} | Acc: {model_metrics['ESM2']['accuracy']:.4f} | ROC-AUC: {model_metrics['ESM2']['roc_auc']:.4f}")

    # 4. Evaluate Calibrated Ensemble (0.70 * ESM-2 + 0.30 * LightGBM)
    print("\n[4/5] Evaluating Calibrated Consensus Ensemble (0.70 * ESM-2 + 0.30 * LightGBM)...")
    prob_lgbm = test_probs["LightGBM"]
    prob_esm2 = test_probs["ESM2"]

    ensemble_probs = 0.70 * prob_esm2 + 0.30 * prob_lgbm
    ensemble_pred = (ensemble_probs >= 0.50).astype(int)
    model_metrics["Ensemble"] = compute_comprehensive_metrics(y_test, ensemble_pred, ensemble_probs)
    print(f"      {'Consensus Ensemble':20s} -> F1: {model_metrics['Ensemble']['f1_score']:.4f} | Acc: {model_metrics['Ensemble']['accuracy']:.4f} | ROC-AUC: {model_metrics['Ensemble']['roc_auc']:.4f}")

    # 5. Compile formatted benchmark summary table and JSON
    print("\n[5/5] Compiling Comprehensive Benchmark Summary Table...")
    
    summary_rows = []
    display_names = {
        "LogisticRegression": "Logistic Regression (Linear L2)",
        "RandomForest": "Random Forest (200 Trees)",
        "XGBoost": "XGBoost (Gradient Boosted Trees)",
        "LightGBM": "LightGBM (Best ML Baseline ⭐)",
        "ESM2": "ESM-2 35M (Protein Transformer 🔬)",
        "Ensemble": "Calibrated Ensemble (0.70 ESM-2 + 0.30 LGBM 🎯)"
    }

    for key, name in display_names.items():
        m = model_metrics[key]
        summary_rows.append({
            "model_key": key,
            "model_name": name,
            "accuracy": round(m["accuracy"] * 100, 2),
            "balanced_accuracy": round(m["balanced_accuracy"] * 100, 2),
            "precision": round(m["precision"] * 100, 2),
            "recall_sensitivity": round(m["recall_sensitivity"] * 100, 2),
            "specificity": round(m["specificity"] * 100, 2),
            "f1_score": round(m["f1_score"] * 100, 2),
            "roc_auc": round(m["roc_auc"] * 100, 2),
            "pr_auc": round(m["pr_auc"] * 100, 2),
            "true_negatives": m["confusion_matrix"]["TN"],
            "false_positives": m["confusion_matrix"]["FP"],
            "false_negatives": m["confusion_matrix"]["FN"],
            "true_positives": m["confusion_matrix"]["TP"],
        })

    out_file = os.path.join(results_dir, "all_models_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(summary_rows, f, indent=2)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(results_dir, "all_models_benchmark.csv"), index=False)

    print("\n" + "=" * 115)
    print(f"{'Model':<40} | {'Acc %':<7} | {'Bal Acc%':<8} | {'Prec %':<7} | {'Recall%':<7} | {'Spec %':<7} | {'F1 %':<7} | {'ROC-AUC':<7} | {'PR-AUC':<7}")
    print("-" * 115)
    for r in summary_rows:
        print(f"{r['model_name']:<40} | {r['accuracy']:<7.2f} | {r['balanced_accuracy']:<8.2f} | {r['precision']:<7.2f} | {r['recall_sensitivity']:<7.2f} | {r['specificity']:<7.2f} | {r['f1_score']:<7.2f} | {r['roc_auc']:<7.2f} | {r['pr_auc']:<7.2f}")
    print("=" * 115)
    print(f"\n[SUCCESS] Benchmark summary saved to {out_file} and results/metrics/all_models_benchmark.csv\n")

if __name__ == "__main__":
    main()
