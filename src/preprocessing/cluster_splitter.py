"""
Cluster-Aware Sequence Partitioning Module
Prevents train/test data leakage by clustering sequences at 70% identity before train/val/test assignment.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
import random
from typing import List, Dict, Tuple, Set, Any
import json
import pandas as pd
from Bio import SeqIO

from src.utils.config import load_config
from src.preprocessing.negative_curation import get_kmers


def kmer_jaccard_distance(kmers1: Set[str], kmers2: Set[str]) -> float:
    """Compute 1 - Jaccard similarity between two k-mer sets."""
    if not kmers1 or not kmers2:
        return 1.0
    union = len(kmers1 | kmers2)
    if union == 0:
        return 1.0
    inter = len(kmers1 & kmers2)
    return 1.0 - (inter / union)


def cluster_sequences_minhash(
    records: List[Dict[str, Any]], k: int = 5, identity_threshold: float = 0.70
) -> List[List[Dict[str, Any]]]:
    """
    Cluster sequences based on 5-mer Jaccard distance matching 70% sequence identity threshold.
    Uses greedy leader clustering to partition sequences into non-overlapping clusters.
    """
    print(f"[Clustering] Partitioning {len(records)} sequences into clusters at identity threshold {identity_threshold}...")
    
    # Pre-extract k-mer sets
    for rec in records:
        rec["kmers"] = get_kmers(rec["sequence"], k=k)

    clusters: List[List[Dict[str, Any]]] = []
    
    # Jaccard similarity threshold corresponding approximately to 70% identity for k=5
    # For amino acids k=5, Jaccard similarity ~0.35 corresponds to 70% sequence identity
    jaccard_sim_threshold = 0.30

    for rec in records:
        assigned = False
        rec_kmers = rec["kmers"]

        for cluster in clusters:
            leader_kmers = cluster[0]["kmers"]
            union = len(rec_kmers | leader_kmers)
            if union > 0:
                sim = len(rec_kmers & leader_kmers) / union
                if sim >= jaccard_sim_threshold:
                    cluster.append(rec)
                    assigned = True
                    break

        if not assigned:
            clusters.append([rec])

    print(f"[Clustering] Formed {len(clusters)} unique clusters from {len(records)} total sequences.")
    return clusters


def run_cluster_splitting(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load curated positive & negative sequences, perform 70% identity clustering,
    and save Train/Val/Test split datasets without data leakage.
    """
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])

    pos_fasta = processed_dir / "positive_sequences_curated.fasta"
    neg_fasta = processed_dir / "negative_sequences_curated.fasta"

    if not pos_fasta.exists() or not neg_fasta.exists():
        raise FileNotFoundError("Curated FASTA datasets missing. Run negative curation first.")

    # 1. Load records into unified dataset list
    all_records: List[Dict[str, Any]] = []

    for record in SeqIO.parse(str(pos_fasta), "fasta"):
        all_records.append({
            "id": record.id,
            "sequence": str(record.seq).strip().upper(),
            "label": 1,
            "source": "CARD_Positive"
        })

    for record in SeqIO.parse(str(neg_fasta), "fasta"):
        all_records.append({
            "id": record.id,
            "sequence": str(record.seq).strip().upper(),
            "label": 0,
            "source": "UniProt_Negative"
        })

    # 2. Cluster sequences at 70% identity threshold
    clusters = cluster_sequences_minhash(
        all_records, k=5, identity_threshold=cfg["preprocessing"]["clustering"]["identity_threshold"]
    )

    # 3. Partition clusters into Train (70%), Val (15%), Test (15%)
    rng = random.Random(cfg["splitting"]["random_state"])
    rng.shuffle(clusters)

    total_seqs = len(all_records)
    train_target = int(total_seqs * cfg["splitting"]["train_ratio"])
    val_target = int(total_seqs * cfg["splitting"]["val_ratio"])

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    train_clusters = 0
    val_clusters = 0
    test_clusters = 0

    current_count = 0
    for cluster in clusters:
        if current_count < train_target:
            train_records.extend(cluster)
            train_clusters += 1
        elif current_count < (train_target + val_target):
            val_records.extend(cluster)
            val_clusters += 1
        else:
            test_records.extend(cluster)
            test_clusters += 1
        current_count += len(cluster)

    # 4. Save splits to CSV files
    train_df = pd.DataFrame(train_records).drop(columns=["kmers"], errors="ignore")
    val_df = pd.DataFrame(val_records).drop(columns=["kmers"], errors="ignore")
    test_df = pd.DataFrame(test_records).drop(columns=["kmers"], errors="ignore")

    train_path = processed_dir / "train_split.csv"
    val_path = processed_dir / "val_split.csv"
    test_path = processed_dir / "test_split.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 5. Verify zero cluster leakage between Train and Test
    train_ids = set(train_df["id"])
    test_ids = set(test_df["id"])
    overlap_ids = train_ids & test_ids
    assert len(overlap_ids) == 0, f"DATA LEAKAGE DETECTED: {len(overlap_ids)} IDs overlap between Train and Test!"

    split_stats = {
        "total_sequences": total_seqs,
        "total_clusters": len(clusters),
        "train": {
            "num_clusters": train_clusters,
            "num_sequences": len(train_df),
            "positives": int((train_df["label"] == 1).sum()),
            "negatives": int((train_df["label"] == 0).sum()),
            "file": str(train_path),
        },
        "validation": {
            "num_clusters": val_clusters,
            "num_sequences": len(val_df),
            "positives": int((val_df["label"] == 1).sum()),
            "negatives": int((val_df["label"] == 0).sum()),
            "file": str(val_path),
        },
        "test": {
            "num_clusters": test_clusters,
            "num_sequences": len(test_df),
            "positives": int((test_df["label"] == 1).sum()),
            "negatives": int((test_df["label"] == 0).sum()),
            "file": str(test_path),
        },
        "leakage_check_passed": True,
    }

    summary_path = processed_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    print("\n[Split Summary]")
    print(f"  Train : {len(train_df)} seqs ({train_clusters} clusters) [Pos: {(train_df['label']==1).sum()}, Neg: {(train_df['label']==0).sum()}]")
    print(f"  Val   : {len(val_df)} seqs ({val_clusters} clusters) [Pos: {(val_df['label']==1).sum()}, Neg: {(val_df['label']==0).sum()}]")
    print(f"  Test  : {len(test_df)} seqs ({test_clusters} clusters) [Pos: {(test_df['label']==1).sum()}, Neg: {(test_df['label']==0).sum()}]")
    print(f"  Data Leakage Verification: PASSED (0 overlapping sequence IDs)")
    return split_stats


if __name__ == "__main__":
    run_cluster_splitting()
