"""
Stage 11: Multi-Class Drug Resistance Family Classifier
Expands binary AMR detection to multi-class prediction of specific drug resistance families:
  1. Beta-lactam (Penicillins, Cephalosporins, Carbapenems, Monobactams)
  2. Aminoglycoside
  3. Fluoroquinolone
  4. Tetracycline
  5. Colistin / Polymyxin (Peptide Antibiotics)
  6. Macrolide
  7. Glycopeptide
  8. Other AMR Determinants
  9. Non-AMR Bacterial Housekeeping Control
"""

import sys
import os

# Crucial Windows fix: Import LightGBM first before numpy/pandas/sklearn
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import load_config


def map_aro_to_drug_family(drug_str: str, model_name: str = "") -> str:
    """Map CARD ontology drug class strings and model names to standardized drug families."""
    if pd.isna(drug_str):
        return "Other"

    d = str(drug_str).lower()
    m = str(model_name).lower()

    # 1. Colistin / Polymyxin (mcr genes, PhoP/Q, pmr mutants, or peptide antibiotics)
    if "colistin" in m or "polymyxin" in m or "mcr-" in m or ("peptide antibiotic" in d and "glycopeptide" not in d):
        return "Colistin"

    # 2. Beta-lactams (penicillins, cephalosporins, carbapenems, monobactams)
    if any(k in d for k in ["beta-lactam", "penicillin", "cephalosporin", "carbapenem", "monobactam"]):
        return "Beta-lactam"

    # 3. Aminoglycosides
    if "aminoglycoside" in d:
        return "Aminoglycoside"

    # 4. Fluoroquinolones / Quinolones
    if "fluoroquinolone" in d or "quinolone" in d:
        return "Fluoroquinolone"

    # 5. Tetracyclines
    if "tetracycline" in d:
        return "Tetracycline"

    # 6. Glycopeptides (vancomycin, teicoplanin, etc.)
    if "glycopeptide" in d:
        return "Glycopeptide"

    # 7. Macrolides
    if "macrolide" in d:
        return "Macrolide"

    return "Other"


def train_multiclass_model():
    print("=" * 80)
    print(" STAGE 11: MULTI-CLASS DRUG RESISTANCE FAMILY CLASSIFIER")
    print("=" * 80)

    cfg = load_config("configs/config.yaml")
    processed_dir = "data/processed"
    models_dir = "models"
    metrics_dir = "results/metrics"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Load CARD ARO metadata for drug class mapping
    print("\n[1/5] Loading CARD ARO categories metadata...")
    aro_cat_path = "data/raw/card/aro_categories_index.tsv"
    aro_df = pd.read_csv(aro_cat_path, sep="\t")
    
    aro_index_path = "data/raw/card/aro_index.tsv"
    aro_index = pd.read_csv(aro_index_path, sep="\t")
    acc_to_model_name = dict(zip(aro_index["Protein Accession"].dropna(), aro_index["Model Name"].dropna()))

    # Build accession -> drug family dictionary
    accession_to_family = {}
    for _, row in aro_df.iterrows():
        acc = str(row["Protein Accession"])
        model_nm = acc_to_model_name.get(acc, "")
        family = map_aro_to_drug_family(row["Drug Class"], model_nm)
        accession_to_family[acc] = family

    # 2. Load Splits and Assign Labels
    print("\n[2/5] Assigning multi-class drug family labels across splits...")
    train_df = pd.read_csv(os.path.join(processed_dir, "train_split.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_split.csv"))

    def get_family_for_row(row):
        if row["label"] == 0:
            return "Non-AMR"
        parts = str(row["id"]).split("|")
        acc = parts[1] if len(parts) > 1 else str(row["id"])
        return accession_to_family.get(acc, "Other")

    train_df["drug_family"] = train_df.apply(get_family_for_row, axis=1)
    test_df["drug_family"] = test_df.apply(get_family_for_row, axis=1)

    classes = [
        "Beta-lactam",
        "Aminoglycoside",
        "Fluoroquinolone",
        "Tetracycline",
        "Colistin",
        "Macrolide",
        "Glycopeptide",
        "Other",
        "Non-AMR",
    ]
    present_classes = [c for c in classes if c in train_df["drug_family"].unique()]
    class_to_idx = {cls: idx for idx, cls in enumerate(present_classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(present_classes)}

    print(f"      Total Target Classes: {len(present_classes)}")
    print("      Class Distribution in Training Set:")
    for cls in present_classes:
        count = sum(train_df["drug_family"] == cls)
        print(f"        - {cls:<20}: {count:>5} sequences")

    y_train = np.ascontiguousarray(train_df["drug_family"].map(class_to_idx).values, dtype=np.float32)
    y_test = np.ascontiguousarray(test_df["drug_family"].map(class_to_idx).values, dtype=np.float32)

    # 3. Load Feature Matrices
    print("\n[3/5] Loading pre-extracted hybrid features (4,000 k-mers + 6 biophysical)...")
    kmer_npz = np.load(os.path.join(processed_dir, "kmer_features.npz"))
    bio_npz = np.load(os.path.join(processed_dir, "biophysical_features.npz"))

    X_train = np.ascontiguousarray(np.hstack([kmer_npz["X_train"], bio_npz["X_train"]]), dtype=np.float32)
    X_test = np.ascontiguousarray(np.hstack([kmer_npz["X_test"], bio_npz["X_test"]]), dtype=np.float32)

    print(f"      X_train shape: {X_train.shape}")
    print(f"      X_test shape:  {X_test.shape}")

    # 4. Train LightGBM Multi-Class Classifier
    print("\n[4/5] Training LightGBM Multi-Class Model (150 trees)...")
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(present_classes),
        n_estimators=150,
        learning_rate=0.05,
        max_depth=7,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    print("      Training complete!")

    # 5. Evaluate on Independent Test Set
    print("\n[5/5] Evaluating on Unseen 70% Identity Test Set (1,137 sequences)...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    accuracy = float(np.mean(y_test == y_pred))

    target_names = [idx_to_class[i] for i in range(len(present_classes))]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    # Save Model & Metrics
    model_artifact = {
        "model": clf,
        "classes": present_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": len(present_classes),
    }
    model_save_path = os.path.join(models_dir, "multiclass_lightgbm.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model_artifact, f)

    metrics_payload = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_report": report_dict,
        "classes": present_classes,
    }
    metrics_save_path = os.path.join(metrics_dir, "multiclass_metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Formatted Console Summary
    print("\n" + "=" * 80)
    print(" STAGE 11 MULTI-CLASS PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")
    print(f"  Weighted F1-Score: {weighted_f1:.4f}")
    print(f"  Macro Precision  : {macro_precision:.4f}")
    print(f"  Macro Recall     : {macro_recall:.4f}")
    print("-" * 80)
    print(f"{'Drug Resistance Family':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Test Support':<10}")
    print("-" * 80)
    for cls in present_classes:
        metrics = report_dict[cls]
        print(f"{cls:<22} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1-score']:<10.4f} | {int(metrics['support']):<10}")
    print("=" * 80)
    print(f"[OK] Multi-class model saved to: {model_save_path}")
    print(f"[OK] Multi-class metrics saved to: {metrics_save_path}\n")

    return metrics_payload


if __name__ == "__main__":
    train_multiclass_model()
