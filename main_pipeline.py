"""
Master Pipeline Orchestrator for AMR Gene Detection & Explainable AI
Executes all stages sequentially with strict logging, metrics collection, and data leakage checks.
"""

import lightgbm as lgb

from pathlib import Path
import sys
import time

from src.utils.config import load_config
from src.data.pipeline_data import run_data_pipeline
from src.preprocessing.sequence_validator import run_data_validation
from src.preprocessing.negative_curation import curate_negative_dataset
from src.preprocessing.cluster_splitter import run_cluster_splitting
from src.features.kmer_extractor import extract_and_save_kmer_features
from src.features.physicochemical import extract_and_save_biophysical_features
from src.models.baseline_models import train_and_evaluate_baselines
from src.explainability.residue_attribution import run_explainability_audit


def run_full_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("AMR GENE DETECTION & EXPLAINABLE AI — MASTER PIPELINE EXECUTION")
    print("=" * 80)

    # 1. Config Check
    print("\n>>> STAGE 1: Configuration Verification...")
    cfg = load_config()
    print(f"    Project: {cfg['project']['name']} | Seed: {cfg['project']['seed']}")

    # 2. Data Acquisition
    print("\n>>> STAGE 2: Dataset Acquisition (CARD, UniProt, AMRFinder)...")
    run_data_pipeline()

    # 3. Data Validation & Quality Control
    print("\n>>> STAGE 3: Sequence Validation & Data Quality Control...")
    run_data_validation()

    # 4. Negative Dataset Curation & Screening
    print("\n>>> STAGE 4: Negative Control Screening & Length Distribution Matching...")
    curate_negative_dataset()

    # 5. Cluster-Based Split (70% Sequence Identity Threshold)
    print("\n>>> STAGE 5: Cluster-Based Train/Val/Test Partitioning (Zero Data Leakage)...")
    run_cluster_splitting()

    # 6. Feature Extraction
    print("\n>>> STAGE 6: Feature Representation (k-mer TF-IDF + Biophysical Properties)...")
    extract_and_save_kmer_features()
    extract_and_save_biophysical_features()

    # 7. Baseline Model Training & Evaluation
    print("\n>>> STAGE 7: Baseline Machine Learning Training & Benchmark Evaluation...")
    baseline_results = train_and_evaluate_baselines()

    # 8. Explainable AI & Biological Attribution Audit
    print("\n>>> STAGE 8: Explainable AI (Integrated Gradients & In-Silico Alanine Scan)...")
    run_explainability_audit()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETE IN {elapsed / 60:.2f} MINUTES")
    print("All outputs, metrics, and models saved under data/, models/, and results/")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
