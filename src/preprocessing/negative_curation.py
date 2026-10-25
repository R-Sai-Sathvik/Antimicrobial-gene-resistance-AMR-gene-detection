"""
Negative Dataset Screening & Length Distribution Matching Module
Ensures negative control sequences are non-resistant, non-homologous to CARD, and length-matched.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
import random
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from src.utils.config import load_config
from src.preprocessing.sequence_validator import SequenceValidator


def get_kmers(seq: str, k: int = 4) -> Set[str]:
    """Extract set of unique k-mers from sequence."""
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def compute_kmer_jaccard(seq1: str, seq2: str, k: int = 4) -> float:
    """Compute Jaccard similarity of k-mer sets between two sequences."""
    kmers1 = get_kmers(seq1, k)
    kmers2 = get_kmers(seq2, k)
    if not kmers1 or not kmers2:
        return 0.0
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return intersection / union if union > 0 else 0.0


def curate_negative_dataset(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Filter candidate negative sequences against positive CARD sequences and perform length matching.

    Returns:
        Dict[str, Any]: Summary statistics of curated positive and negative datasets.
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    validator = SequenceValidator(
        min_len=cfg["preprocessing"]["min_length"],
        max_len=cfg["preprocessing"]["max_length"],
    )

    card_fasta = raw_dir / "card" / cfg["data"]["card"]["fasta_filename"]
    uniprot_fasta = raw_dir / "uniprot" / "uniprot_bacterial_candidates.fasta"

    if not card_fasta.exists() or not uniprot_fasta.exists():
        raise FileNotFoundError("Raw CARD or UniProt FASTA files missing. Run dataset download first.")

    # 1. Parse and validate positive CARD sequences
    print("[Negative Curation] Loading and validating CARD positive sequences...")
    positive_records: List[SeqRecord] = []
    card_sequence_set: Set[str] = set()
    card_kmer_dict: Dict[str, Set[str]] = {}

    for record in SeqIO.parse(str(card_fasta), "fasta"):
        seq_str = str(record.seq).strip().upper()
        is_valid, _ = validator.validate_single_sequence(seq_str)
        if is_valid:
            positive_records.append(record)
            card_sequence_set.add(seq_str)
            card_kmer_dict[record.id] = get_kmers(seq_str, k=4)

    print(f"[Negative Curation] Validated positive CARD sequences: {len(positive_records)}")

    # 2. Filter UniProt candidate negatives against CARD
    print("[Negative Curation] Screening UniProt candidates against CARD positives...")
    screened_negative_records: List[SeqRecord] = []
    excluded_exact_matches = 0
    excluded_high_sim = 0

    # Build k-mer index for fast filtering
    card_kmer_union = set().union(*card_kmer_dict.values())

    for record in SeqIO.parse(str(uniprot_fasta), "fasta"):
        seq_str = str(record.seq).strip().upper()
        is_valid, _ = validator.validate_single_sequence(seq_str)
        if not is_valid:
            continue

        # Exact sequence match check
        if seq_str in card_sequence_set:
            excluded_exact_matches += 1
            continue

        # Fast k-mer containment score check
        cand_kmers = get_kmers(seq_str, k=4)
        if not cand_kmers:
            continue

        shared_ratio = len(cand_kmers & card_kmer_union) / len(cand_kmers)
        # Exclude if candidate sequence has > 85% k-mer overlap with CARD universe
        if shared_ratio > 0.85:
            excluded_high_sim += 1
            continue

        screened_negative_records.append(record)

    print(f"[Negative Curation] Screened negatives remaining: {len(screened_negative_records)}")
    print(f"[Negative Curation] Excluded exact matches: {excluded_exact_matches}, high-similarity: {excluded_high_sim}")

    # 3. Length distribution binning and balanced sampling
    print("[Negative Curation] Matching length distributions between positive and negative sets...")
    bins = [50, 150, 300, 450, 600, 800, 1200, 2500]

    def get_len_bin(l: int) -> int:
        for i, b in enumerate(bins[1:]):
            if l <= b:
                return i
        return len(bins) - 2

    pos_binned = defaultdict(list)
    for rec in positive_records:
        pos_binned[get_len_bin(len(rec.seq))].append(rec)

    neg_binned = defaultdict(list)
    for rec in screened_negative_records:
        neg_binned[get_len_bin(len(rec.seq))].append(rec)

    # Sample negatives per bin (target 2x positive count per bin or max available)
    rng = random.Random(cfg["project"]["seed"])
    curated_negative_records: List[SeqRecord] = []

    for bin_idx, pos_recs in pos_binned.items():
        target_count = min(len(pos_recs) * 2, len(neg_binned[bin_idx]))
        if target_count > 0:
            sampled_negs = rng.sample(neg_binned[bin_idx], target_count)
            curated_negative_records.extend(sampled_negs)

    print(f"[Negative Curation] Final curated negative count: {len(curated_negative_records)}")

    # 4. Save curated FASTA datasets
    pos_out = processed_dir / "positive_sequences_curated.fasta"
    neg_out = processed_dir / "negative_sequences_curated.fasta"

    SeqIO.write(positive_records, str(pos_out), "fasta")
    SeqIO.write(curated_negative_records, str(neg_out), "fasta")

    stats = {
        "positive_count": len(positive_records),
        "screened_negative_count": len(screened_negative_records),
        "final_curated_negative_count": len(curated_negative_records),
        "positive_fasta": str(pos_out),
        "negative_fasta": str(neg_out),
        "excluded_exact_matches": excluded_exact_matches,
        "excluded_high_kmer_similarity": excluded_high_sim,
    }

    summary_file = processed_dir / "curation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        import json
        json.dump(stats, f, indent=2)

    print(f"[Negative Curation] Curated datasets saved to:\n  Positives: {pos_out}\n  Negatives: {neg_out}")
    return stats


if __name__ == "__main__":
    curate_negative_dataset()
