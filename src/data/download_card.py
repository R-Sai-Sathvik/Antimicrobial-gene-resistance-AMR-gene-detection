"""
CARD Database Downloader & Provenance Manager
Automates the retrieval and extraction of the Comprehensive Antibiotic Resistance Database (CARD).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
import hashlib
import tarfile
import time
from typing import Dict, Any
import requests
from Bio import SeqIO

from src.utils.config import load_config


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a local file."""
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_card_database(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Download and extract the latest CARD reference dataset.

    Returns:
        Dict[str, Any]: Metadata summary of the downloaded dataset.
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"]) / "card"
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = cfg["data"]["card"]["download_url"]
    archive_path = raw_dir / cfg["data"]["card"]["archive_filename"]

    print(f"[CARD Download] Downloading latest CARD archive from: {url}")
    headers = {"User-Agent": "AMR-ML-Research-Pipeline/1.0"}
    
    # Download tarball
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    start_time = time.time()
    total_bytes = 0
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    download_time = time.time() - start_time
    file_sha256 = compute_file_hash(archive_path, "sha256")
    print(f"[CARD Download] Downloaded {total_bytes / (1024*1024):.2f} MB in {download_time:.1f}s (SHA256: {file_sha256[:12]}...)")

    # Extract tarball
    print(f"[CARD Download] Extracting tarball to {raw_dir}...")
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=raw_dir)

    fasta_path = raw_dir / cfg["data"]["card"]["fasta_filename"]
    aro_path = raw_dir / cfg["data"]["card"]["aro_index_filename"]

    if not fasta_path.exists():
        raise FileNotFoundError(f"Expected fasta file not found after extraction: {fasta_path}")

    # Count sequence records
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    num_sequences = len(records)

    metadata = {
        "source": "CARD (Comprehensive Antibiotic Resistance Database)",
        "download_url": url,
        "archive_filename": cfg["data"]["card"]["archive_filename"],
        "archive_size_bytes": total_bytes,
        "sha256": file_sha256,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "fasta_file": str(fasta_path),
        "aro_file": str(aro_path),
        "num_sequences": num_sequences,
    }

    # Record provenance in data/raw/README.md
    readme_path = Path(cfg["paths"]["raw_dir"]) / "README.md"
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write(f"\n## CARD Database Provenance\n")
        f.write(f"- **Source**: {metadata['source']}\n")
        f.write(f"- **URL**: {metadata['download_url']}\n")
        f.write(f"- **Downloaded UTC**: {metadata['timestamp_utc']}\n")
        f.write(f"- **SHA256**: `{metadata['sha256']}`\n")
        f.write(f"- **Sequence Count**: {metadata['num_sequences']} protein sequences\n")
        f.write(f"- **Fasta Path**: `{metadata['fasta_file']}`\n")

    print(f"[CARD Download] Successfully processed {num_sequences} CARD protein homolog sequences.")
    return metadata


if __name__ == "__main__":
    download_card_database()
