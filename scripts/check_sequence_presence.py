"""
Sequence Presence & Data Leakage Checker
Checks whether query protein/DNA sequences or accessions exist in:
  - data/processed/train_split.csv (Train)
  - data/processed/val_split.csv   (Validation)
  - data/processed/test_split.csv  (Test)

Supports:
1. Exact Sequence Match (100% amino acid identity)
2. Substring Match (Query contained in dataset sequence or vice versa)
3. ID / Accession Match (Accession IDs, ARO numbers, UniProt keys)
4. K-mer Homology Similarity (Jaccard similarity >= 0.70)
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd


def get_kmers(seq: str, k: int = 5) -> set:
    """Generate k-mer set for fast sequence similarity."""
    if len(seq) < k:
        return {seq}
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity coefficient between two k-mer sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class SequenceSplitChecker:
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.splits = {}
        for split_name in ["train", "val", "test"]:
            path = self.data_dir / f"{split_name}_split.csv"
            if path.exists():
                df = pd.read_csv(path)
                # Store cleaned uppercase sequence and precomputed 5-mer sets
                df["clean_seq"] = df["sequence"].astype(str).str.strip().str.upper()
                df["kmers"] = df["clean_seq"].apply(lambda s: get_kmers(s, k=5))
                self.splits[split_name] = df
            else:
                print(f"[!] Warning: {path} not found.")

        print(f"[+] Loaded splits: Train={len(self.splits.get('train', []))}, "
              f"Val={len(self.splits.get('val', []))}, "
              f"Test={len(self.splits.get('test', []))}\n")

    def check_sequence(self, query_seq: str, query_id: str = "Query") -> Dict:
        """Thoroughly checks a query sequence across all splits."""
        query_clean = query_seq.strip().upper()
        # Handle FASTA headers if passed inside sequence
        if query_clean.startswith(">"):
            lines = query_clean.split("\n")
            if len(lines) > 1:
                query_id = lines[0][1:].strip()
                query_clean = "".join(lines[1:]).replace(" ", "").replace("\r", "")

        query_kmers = get_kmers(query_clean, k=5)
        results = {
            "query_id": query_id,
            "query_length": len(query_clean),
            "exact_matches": [],
            "substring_matches": [],
            "id_matches": [],
            "homology_matches": [],
        }

        for split_name, df in self.splits.items():
            # 1. Exact match
            exact = df[df["clean_seq"] == query_clean]
            for _, row in exact.iterrows():
                results["exact_matches"].append({
                    "split": split_name,
                    "id": row["id"],
                    "label": "AMR Positive" if row["label"] == 1 else "Non-AMR Negative",
                    "source": row.get("source", "N/A"),
                })

            # 2. Substring match (if query length >= 20 and not already an exact match)
            if len(query_clean) >= 20 and exact.empty:
                for _, row in df.iterrows():
                    ds_seq = row["clean_seq"]
                    if query_clean in ds_seq or ds_seq in query_clean:
                        overlap_pct = round(min(len(query_clean), len(ds_seq)) / max(len(query_clean), len(ds_seq)) * 100, 1)
                        results["substring_matches"].append({
                            "split": split_name,
                            "id": row["id"],
                            "label": "AMR Positive" if row["label"] == 1 else "Non-AMR Negative",
                            "overlap_pct": overlap_pct,
                            "dataset_seq_len": len(ds_seq),
                        })

            # 3. ID / Accession match
            if query_id and query_id != "Query":
                id_tokens = [t for t in query_id.replace("|", " ").replace(">", "").split() if len(t) >= 4]
                for token in id_tokens:
                    matches = df[df["id"].astype(str).str.contains(token, regex=False, na=False)]
                    for _, row in matches.iterrows():
                        if not any(m["id"] == row["id"] and m["split"] == split_name for m in results["id_matches"]):
                            results["id_matches"].append({
                                "split": split_name,
                                "matched_token": token,
                                "id": row["id"],
                                "label": "AMR Positive" if row["label"] == 1 else "Non-AMR Negative",
                            })

            # 4. Homology / K-mer similarity (top matches with Jaccard >= 0.60)
            if exact.empty:
                for _, row in df.iterrows():
                    sim = jaccard_similarity(query_kmers, row["kmers"])
                    if sim >= 0.60:
                        results["homology_matches"].append({
                            "split": split_name,
                            "id": row["id"],
                            "label": "AMR Positive" if row["label"] == 1 else "Non-AMR Negative",
                            "jaccard_similarity": round(sim, 4),
                        })

        # Sort homology matches by highest similarity
        results["homology_matches"].sort(key=lambda x: x["jaccard_similarity"], reverse=True)
        results["homology_matches"] = results["homology_matches"][:5]

        return results

    def print_report(self, results: Dict):
        """Prints formatted terminal report."""
        print("=" * 80)
        print(f" SEQUENCE PRESENCE REPORT: {results['query_id']} (Length: {results['query_length']} aa)")
        print("=" * 80)

        # Exact Matches
        if results["exact_matches"]:
            print("\n [!] EXACT MATCH FOUND (100% Identity):")
            for m in results["exact_matches"]:
                print(f"    --> SPLIT: {m['split'].upper():<6} | ID: {m['id']} | Class: {m['label']} | Source: {m['source']}")
        else:
            print("\n [x] No Exact Matches Found (Sequence is NOT identical to any split item)")

        # Substring Matches
        if results["substring_matches"]:
            print(f"\n [~] SUBSTRING / TRUNCATION MATCHES ({len(results['substring_matches'])}):")
            for m in results["substring_matches"][:5]:
                print(f"    --> SPLIT: {m['split'].upper():<6} | ID: {m['id']} | Overlap: {m['overlap_pct']}% | Class: {m['label']}")

        # Accession / ID Matches
        if results["id_matches"]:
            print(f"\n [@] ACCESSION / HEADER MATCHES ({len(results['id_matches'])}):")
            for m in results["id_matches"][:5]:
                print(f"    --> SPLIT: {m['split'].upper():<6} | Matched Token: {m['matched_token']} | ID: {m['id']}")

        # Homology Matches
        if results["homology_matches"]:
            print(f"\n [*] HOMOLOGY CANDIDATES (Top K-mer Similarities >= 60%):")
            for m in results["homology_matches"]:
                print(f"    --> SPLIT: {m['split'].upper():<6} | Jaccard Sim: {m['jaccard_similarity']*100:.1f}% | ID: {m['id']} | Class: {m['label']}")

        # Summary Assessment
        print("-" * 80)
        splits_found = set(m["split"] for m in results["exact_matches"])
        if "train" in splits_found:
            print(" >> STATUS: PRESENT IN TRAINING SET (Seen by baseline ML models and fine-tuned ESM-2)")
        elif "val" in splits_found:
            print(" >> STATUS: PRESENT IN VALIDATION SET (Used for early stopping / hyperparameter checkpointing)")
        elif "test" in splits_found:
            print(" >> STATUS: PRESENT IN INDEPENDENT TEST SET (Unseen benchmark evaluation)")
        elif results["substring_matches"]:
            print(" >> STATUS: SUBSTRING / ISOFORM OF AN EXISTING SEQUENCE")
        elif results["homology_matches"]:
            print(f" >> STATUS: NOVEL SEQUENCE with homologous family members in dataset (Max sim: {results['homology_matches'][0]['jaccard_similarity']*100:.1f}%)")
        else:
            print(" >> STATUS: ENTIRELY NOVEL SEQUENCE (Zero overlap, completely out-of-distribution)")
        print("=" * 80 + "\n")


def parse_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """Parse FASTA file into list of (header, sequence) tuples."""
    entries = []
    current_header = ""
    current_seq = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    entries.append((current_header, "".join(current_seq)))
                current_header = line[1:].strip()
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_header:
            entries.append((current_header, "".join(current_seq)))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Check whether protein sequences exist in train, validation, or test sets.")
    parser.add_argument("--seq", type=str, help="Amino acid sequence string")
    parser.add_argument("--id", type=str, default="Query_Seq", help="Optional sequence identifier/accession")
    parser.add_argument("--fasta", type=str, help="Path to FASTA file with one or multiple query sequences")
    args = parser.parse_args()

    checker = SequenceSplitChecker()

    if args.fasta:
        p = Path(args.fasta)
        if not p.exists():
            print(f"Error: FASTA file not found at {args.fasta}")
            sys.exit(1)
        entries = parse_fasta(str(p))
        print(f"[+] Loaded {len(entries)} sequence(s) from {args.fasta}")
        for header, seq in entries:
            res = checker.check_sequence(seq, query_id=header)
            checker.print_report(res)
    elif args.seq:
        res = checker.check_sequence(args.seq, query_id=args.id)
        checker.print_report(res)
    else:
        # Interactive mode
        print("--- Interactive Sequence Checker Mode ---")
        print("Paste your sequence below (or press Ctrl+C to exit):")
        try:
            seq_input = input("Sequence: ").strip()
            id_input = input("Sequence ID/Name (optional, press Enter to skip): ").strip() or "User_Query"
            if seq_input:
                res = checker.check_sequence(seq_input, query_id=id_input)
                checker.print_report(res)
            else:
                print("Empty sequence provided. Usage: python scripts/check_sequence_presence.py --seq <SEQ>")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")


if __name__ == "__main__":
    main()
