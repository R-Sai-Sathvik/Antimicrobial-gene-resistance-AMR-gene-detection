"""
NCBI AMRFinderPlus Catalog Downloader
Retrieves the NCBI Reference Gene Catalog for independent external model validation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
import time
from typing import Dict, Any
import requests
import pandas as pd

from src.utils.config import load_config
from src.data.download_card import compute_file_hash


def download_amrfinder_catalog(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Download NCBI AMRFinderPlus Reference Gene Catalog.

    Returns:
        Dict[str, Any]: Download metadata summary.
    """
    cfg = load_config(config_path)
    external_dir = Path(cfg["paths"]["external_dir"]) / "amrfinder"
    external_dir.mkdir(parents=True, exist_ok=True)

    catalog_url = cfg["data"]["amrfinder"]["catalog_url"]
    catalog_path = external_dir / "ReferenceGeneCatalog.txt"

    print(f"[AMRFinder Download] Downloading catalog from: {catalog_url}")
    response = requests.get(catalog_url, stream=True, timeout=60)
    response.raise_for_status()

    start_time = time.time()
    total_bytes = 0
    with open(catalog_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    download_time = time.time() - start_time
    file_sha256 = compute_file_hash(catalog_path, "sha256")

    # Read catalog statistics with pandas
    df = pd.read_csv(catalog_path, sep="\t", low_memory=False)
    num_entries = len(df)
    unique_genes = df["Gene symbol"].nunique() if "Gene symbol" in df.columns else None

    metadata = {
        "source": "NCBI AMRFinderPlus Reference Gene Catalog",
        "url": catalog_url,
        "sha256": file_sha256,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "file_path": str(catalog_path),
        "num_entries": num_entries,
        "unique_gene_symbols": unique_genes,
        "elapsed_seconds": round(download_time, 2),
    }

    readme_path = Path(cfg["paths"]["external_dir"]) / "README.md"
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write(f"\n## AMRFinderPlus Catalog Provenance\n")
        f.write(f"- **Source**: {metadata['source']}\n")
        f.write(f"- **URL**: {metadata['url']}\n")
        f.write(f"- **Downloaded UTC**: {metadata['timestamp_utc']}\n")
        f.write(f"- **SHA256**: `{metadata['sha256']}`\n")
        f.write(f"- **Catalog Entries**: {metadata['num_entries']}\n")

    print(f"[AMRFinder Download] Successfully downloaded {num_entries} catalog entries ({unique_genes} unique symbols).")
    return metadata


if __name__ == "__main__":
    download_amrfinder_catalog()
