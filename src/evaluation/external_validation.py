"""
Stage 10: External Validation and Benchmarking against NCBI AMRFinderPlus
Evaluates trained LightGBM and ESM-2 Transformer models on zero-leakage test sequences
and cross-references with the 11,456-entry NCBI AMRFinderPlus Reference Gene Catalog.
"""

import sys
import os

# Crucial Windows fix: Import LightGBM first and run inference before importing PyTorch
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import load_config
from src.models.baseline_models import compute_comprehensive_metrics


def run_external_validation():
    print("=" * 80)
    print(" STAGE 10: EXTERNAL VALIDATION AGAINST NCBI AMRFINDERPLUS CATALOG")
    print("=" * 80)

    cfg = load_config("configs/config.yaml")
    processed_dir = "data/processed"
    results_dir = "results/metrics"
    models_dir = "models"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load AMRFinder Reference Gene Catalog
    amrfinder_path = "data/external/amrfinder/ReferenceGeneCatalog.txt"
    print(f"\n[1/5] Loading NCBI AMRFinderPlus Reference Gene Catalog from {amrfinder_path}...")
    catalog = pd.read_csv(amrfinder_path, sep="\t", low_memory=False)
    print(f"      Total catalog entries: {len(catalog):,}")
    amr_entries = catalog[catalog["type"] == "AMR"]
    print(f"      AMR resistance entries: {len(amr_entries):,}")
    print(f"      Curated drug classes in catalog: {catalog['class'].nunique()}")

    # 2. Load Independent Test Split
    print(f"\n[2/5] Loading zero-leakage 70% identity test split...")
    test_df = pd.read_csv(os.path.join(processed_dir, "test_split.csv"))
    print(f"      Test set size: {len(test_df)} sequences")
    print(f"      Class distribution: {test_df['label'].value_counts().to_dict()}")

    # 3. Cross-reference test sequences with AMRFinder accessions
    print(f"\n[3/5] Cross-referencing test set AMR genes with AMRFinder accessions...")
    cat_refseq = set(catalog["refseq_protein_accession"].dropna())
    cat_genbank = set(catalog["genbank_protein_accession"].dropna())
    all_cat_accs = cat_refseq | cat_genbank

    def extract_acc(id_str):
        parts = str(id_str).split("|")
        return parts[1] if len(parts) > 1 else str(id_str)

    test_df["accession"] = test_df["id"].apply(extract_acc)
    test_df["in_amrfinder"] = test_df["accession"].isin(all_cat_accs)
    
    # Map AMRFinder class where available
    acc_to_class = dict(zip(catalog["refseq_protein_accession"].dropna(), catalog["class"].dropna()))
    acc_to_class.update(dict(zip(catalog["genbank_protein_accession"].dropna(), catalog["class"].dropna())))
    test_df["amrfinder_class"] = test_df["accession"].map(acc_to_class)

    matched_amr = test_df[(test_df["label"] == 1) & (test_df["in_amrfinder"])]
    print(f"      Test AMR genes matching AMRFinder: {len(matched_amr)} / {sum(test_df['label'] == 1)} ({len(matched_amr)/sum(test_df['label'] == 1)*100:.1f}%)")

    # 4. Evaluate Baseline Model (LightGBM) BEFORE importing PyTorch
    print(f"\n[4/5] Evaluating Baseline LightGBM on independent test features...")
    kmer_npz = np.load(os.path.join(processed_dir, "kmer_features.npz"))
    bio_npz = np.load(os.path.join(processed_dir, "biophysical_features.npz"))
    X_test = np.ascontiguousarray(np.hstack([kmer_npz["X_test"], bio_npz["X_test"]]), dtype=np.float32)
    y_test = np.ascontiguousarray(kmer_npz["y_test"], dtype=np.float32)

    with open(os.path.join(models_dir, "baseline_lightgbm.pkl"), "rb") as f:
        lgbm_model = pickle.load(f)

    y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
    y_pred_lgbm = (y_prob_lgbm >= 0.50).astype(int)
    lgbm_metrics = compute_comprehensive_metrics(y_test, y_pred_lgbm, y_prob_lgbm)
    print(f"      LightGBM Test F1: {lgbm_metrics['f1_score']:.4f} | ROC-AUC: {lgbm_metrics['roc_auc']:.4f}")

    # 5. Evaluate Deep Learning Model (ESM-2 Transformer) - PyTorch imported now
    print(f"\n[5/5] Evaluating ESM-2 35M Protein Transformer on independent test sequences...")
    import torch
    from transformers import AutoTokenizer
    from src.models.esm2_model import ESM2Classifier, ProteinSequenceDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Inference device: {device}")

    model_name = cfg["features"]["esm"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = ESM2Classifier(model_name=model_name).to(device)

    esm_ckpt = os.path.join(models_dir, "esm2_amr_classifier.pt")
    esm_model.load_state_dict(torch.load(esm_ckpt, map_location=device, weights_only=False))
    esm_model.eval()

    test_ds = ProteinSequenceDataset(test_df["sequence"].tolist(), test_df["label"].tolist(), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    y_prob_esm = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, _ = esm_model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob_esm.extend(probs)

    y_prob_esm = np.array(y_prob_esm)
    y_pred_esm = (y_prob_esm >= 0.50).astype(int)
    esm_metrics = compute_comprehensive_metrics(y_test, y_pred_esm, y_prob_esm)
    print(f"      ESM-2 Test F1:    {esm_metrics['f1_score']:.4f} | ROC-AUC: {esm_metrics['roc_auc']:.4f}")

    # Cross-referenced AMRFinder subset evaluation
    amrfinder_mask = test_df["in_amrfinder"].values
    matched_indices = np.where(amrfinder_mask & (y_test == 1))[0]
    
    lgbm_amrfinder_sens = float(np.mean(y_pred_lgbm[matched_indices] == 1))
    esm_amrfinder_sens = float(np.mean(y_pred_esm[matched_indices] == 1))

    # Per-class detection rate in AMRFinder subset
    class_detection = {}
    matched_test_df = test_df.iloc[matched_indices].copy()
    matched_test_df["lgbm_pred"] = y_pred_lgbm[matched_indices]
    matched_test_df["esm_pred"] = y_pred_esm[matched_indices]

    for drug_cls, grp in matched_test_df.groupby("amrfinder_class"):
        if len(grp) >= 5:
            class_detection[drug_cls] = {
                "count": len(grp),
                "lightgbm_sensitivity": float(grp["lgbm_pred"].mean()),
                "esm2_sensitivity": float(grp["esm_pred"].mean()),
            }

    # Compile Final Report
    report = {
        "catalog_summary": {
            "amrfinder_total_entries": len(catalog),
            "amrfinder_amr_entries": len(amr_entries),
            "test_set_size": len(test_df),
            "test_amr_genes_in_amrfinder": len(matched_indices),
            "coverage_percent": round(len(matched_indices) / sum(test_df["label"] == 1) * 100, 2),
        },
        "overall_metrics": {
            "lightgbm": lgbm_metrics,
            "esm2": esm_metrics,
        },
        "amrfinder_confirmed_sensitivity": {
            "lightgbm": lgbm_amrfinder_sens,
            "esm2": esm_amrfinder_sens,
        },
        "per_class_amrfinder_detection": class_detection,
    }

    report_path = os.path.join(results_dir, "external_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Formatted Results
    print("\n" + "=" * 80)
    print(" EXTERNAL BENCHMARKING RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'LightGBM (Baseline)':<22} | {'ESM-2 (Transformer)':<22}")
    print("-" * 80)
    metric_keys = [
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Precision (PPV)", "precision"),
        ("Recall / Sensitivity (TPR)", "recall_sensitivity"),
        ("Specificity (TNR)", "specificity"),
        ("F1-Score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]
    for label, k in metric_keys:
        print(f"{label:<25} | {lgbm_metrics[k]:<22.4f} | {esm_metrics[k]:<22.4f}")

    print("-" * 80)
    print(f"{'AMRFinder Confirmed Sens':<25} | {lgbm_amrfinder_sens*100:<21.2f}% | {esm_amrfinder_sens*100:<21.2f}%")
    print("=" * 80)

    print("\nDetection Sensitivity by AMRFinder Drug Class (Top Classes):")
    print(f"{'Drug Class':<25} | {'Count':<7} | {'LightGBM Sens':<15} | {'ESM-2 Sens':<15}")
    print("-" * 70)
    for drug_cls, data in sorted(class_detection.items(), key=lambda x: -x[1]["count"])[:8]:
        print(f"{drug_cls:<25} | {data['count']:<7} | {data['lightgbm_sensitivity']*100:<14.1f}% | {data['esm2_sensitivity']*100:<14.1f}%")
    print("=" * 70)
    print(f"\n[OK] External validation report saved to: {report_path}\n")

    return report


if __name__ == "__main__":
    run_external_validation()
