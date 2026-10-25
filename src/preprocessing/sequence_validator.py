"""
Sequence Validation & Data Quality Control Module
Enforces strict biological quality control, alphabet checks, length bounds, and duplicate detection.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from collections import Counter
from Bio import SeqIO

from src.utils.config import load_config


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class SequenceValidator:
    """Validates biological protein sequences for machine learning dataset construction."""

    def __init__(self, min_len: int = 50, max_len: int = 2500):
        self.min_len = min_len
        self.max_len = max_len
        self.valid_aa = VALID_AMINO_ACIDS

    def validate_single_sequence(self, seq_str: str) -> Tuple[bool, str]:
        """
        Check single amino acid sequence string for validity.

        Returns:
            Tuple[bool, str]: (is_valid, reason_if_invalid)
        """
        seq_clean = seq_str.strip().upper()
        if len(seq_clean) < self.min_len:
            return False, f"Sequence length {len(seq_clean)} < min_length ({self.min_len})"
        if len(seq_clean) > self.max_len:
            return False, f"Sequence length {len(seq_clean)} > max_length ({self.max_len})"

        invalid_chars = set(seq_clean) - self.valid_aa
        if invalid_chars:
            return False, f"Invalid amino acid characters present: {sorted(list(invalid_chars))}"

        return True, "Valid"

    def audit_fasta_file(self, fasta_path: Path, dataset_name: str) -> Dict[str, Any]:
        """
        Audit a FASTA file for sequence quality, duplicates, and alphabet distribution.

        Returns:
            Dict[str, Any]: Quality control audit metrics.
        """
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        total_records = 0
        valid_records = 0
        short_seqs = 0
        long_seqs = 0
        invalid_char_seqs = 0

        seen_ids = set()
        duplicate_ids = 0
        seen_sequences = dict()  # seq -> list of IDs
        duplicate_sequences = 0

        lengths = []
        aa_counts = Counter()

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            total_records += 1
            seq_id = record.id
            seq_str = str(record.seq).strip().upper()

            # ID duplicate check
            if seq_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(seq_id)

            # Sequence duplicate check
            if seq_str in seen_sequences:
                duplicate_sequences += 1
                seen_sequences[seq_str].append(seq_id)
            else:
                seen_sequences[seq_str] = [seq_id]

            # Length check
            seq_len = len(seq_str)
            lengths.append(seq_len)

            if seq_len < self.min_len:
                short_seqs += 1
                continue
            if seq_len > self.max_len:
                long_seqs += 1
                continue

            # Alphabet check
            invalid_chars = set(seq_str) - self.valid_aa
            if invalid_chars:
                invalid_char_seqs += 1
                continue

            valid_records += 1
            aa_counts.update(seq_str)

        mean_len = sum(lengths) / len(lengths) if lengths else 0
        sorted_lens = sorted(lengths)
        median_len = sorted_lens[len(sorted_lens) // 2] if sorted_lens else 0

        report = {
            "dataset_name": dataset_name,
            "fasta_path": str(fasta_path),
            "total_records": total_records,
            "valid_records": valid_records,
            "pass_rate_percent": round(valid_records / total_records * 100, 2) if total_records else 0,
            "rejections": {
                "short_sequences": short_seqs,
                "long_sequences": long_seqs,
                "invalid_characters": invalid_char_seqs,
            },
            "duplicates": {
                "duplicate_header_ids": duplicate_ids,
                "duplicate_exact_sequences": duplicate_sequences,
                "unique_sequences_count": len(seen_sequences),
            },
            "length_stats": {
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "mean_length": round(mean_len, 2),
                "median_length": median_len,
            },
            "amino_acid_composition": dict(aa_counts),
        }
        return report


def run_data_validation(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Run full data quality validation audit on raw datasets."""
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

    reports = {}
    if card_fasta.exists():
        print(f"[Validation] Auditing CARD dataset: {card_fasta}")
        reports["card"] = validator.audit_fasta_file(card_fasta, "CARD Positive Database")

    if uniprot_fasta.exists():
        print(f"[Validation] Auditing UniProt candidate negatives dataset: {uniprot_fasta}")
        reports["uniprot"] = validator.audit_fasta_file(uniprot_fasta, "UniProt Candidate Negatives")

    report_path = processed_dir / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"[Validation] Validation complete. Data quality report saved to: {report_path}")
    return reports


if __name__ == "__main__":
    run_data_validation()
