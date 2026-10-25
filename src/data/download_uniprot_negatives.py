"""
UniProt Negative Sequence Candidate Downloader
Retrieves high-confidence Swiss-Prot reviewed non-AMR bacterial protein sequences.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
import time
from typing import Dict, Any
import requests
from Bio import SeqIO

from src.utils.config import load_config
from src.data.download_card import compute_file_hash


def download_uniprot_negative_candidates(
    max_sequences: int = 12000, config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Query UniProt REST API for reviewed (Swiss-Prot) bacterial proteins excluding AMR keywords.

    Args:
        max_sequences (int): Target maximum number of candidate negative sequences.
        config_path (str): Path to config YAML.

    Returns:
        Dict[str, Any]: Download metadata.
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"]) / "uniprot"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_fasta = raw_dir / "uniprot_bacterial_candidates.fasta"

    # UniProt search query
    query = (
        "(taxonomy_id:2) AND (reviewed:true) "
        "NOT (keyword:KW-0046) NOT (beta-lactamase) NOT (transporter) "
        "NOT (efflux) NOT (penicillin-binding) NOT (acetyltransferase)"
    )

    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "fasta",
        "query": query,
        "size": 500,
    }

    print(f"[UniProt Download] Querying UniProtKB stream API...")
    print(f"[UniProt Download] Query: {query}")

    start_time = time.time()
    response = requests.get(url, params=params, stream=True, timeout=120)
    response.raise_for_status()

    total_bytes = 0
    record_count = 0

    with open(output_fasta, "w", encoding="utf-8") as out_f:
        chunk_lines = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                chunk_lines.append(line + "\n")
                total_bytes += len(line)
                if line.startswith(">"):
                    record_count += 1
                    if record_count % 2000 == 0:
                        print(f"[UniProt Download] Downloaded {record_count} candidate negative sequences...")

                if record_count >= max_sequences and line.startswith(">") and len(chunk_lines) > 1:
                    # Write up to target count
                    out_f.writelines(chunk_lines[:-1])
                    break

                if len(chunk_lines) >= 1000:
                    out_f.writelines(chunk_lines)
                    chunk_lines = []

        if chunk_lines and record_count < max_sequences:
            out_f.writelines(chunk_lines)

    download_time = time.time() - start_time
    file_sha256 = compute_file_hash(output_fasta, "sha256")

    # Verify extracted records with Biopython
    valid_records = list(SeqIO.parse(str(output_fasta), "fasta"))
    actual_count = len(valid_records)

    metadata = {
        "source": "UniProtKB / Swiss-Prot (Bacterial Reviewed)",
        "download_url": url,
        "query": query,
        "sha256": file_sha256,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "fasta_file": str(output_fasta),
        "num_sequences": actual_count,
        "elapsed_seconds": round(download_time, 2),
    }

    readme_path = Path(cfg["paths"]["raw_dir"]) / "README.md"
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write(f"\n## UniProt Negative Candidates Provenance\n")
        f.write(f"- **Source**: {metadata['source']}\n")
        f.write(f"- **Query**: `{metadata['query']}`\n")
        f.write(f"- **Downloaded UTC**: {metadata['timestamp_utc']}\n")
        f.write(f"- **SHA256**: `{metadata['sha256']}`\n")
        f.write(f"- **Sequence Count**: {metadata['num_sequences']} protein sequences\n")
        f.write(f"- **Fasta Path**: `{metadata['fasta_file']}`\n")

    print(f"[UniProt Download] Successfully saved {actual_count} candidate negative sequences in {download_time:.1f}s.")
    return metadata


if __name__ == "__main__":
    download_uniprot_negative_candidates()
