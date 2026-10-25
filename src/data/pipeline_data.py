"""
Data Pipeline Orchestrator
Executes dataset downloads (CARD, UniProt, AMRFinderPlus) and logs dataset inventory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
import json
import time

from src.data.download_card import download_card_database
from src.data.download_uniprot_negatives import download_uniprot_negative_candidates
from src.data.download_amrfinder import download_amrfinder_catalog
from src.utils.config import load_config


def run_data_pipeline(config_path: str = "configs/config.yaml") -> None:
    """Run full dataset download and inventory pipeline."""
    cfg = load_config(config_path)
    print("=" * 70)
    print("STAGE 2: DATASET ACQUISITION PIPELINE")
    print("=" * 70)

    # 1. Download CARD database
    print("\n--- Step 1: Downloading CARD Database ---")
    card_meta = download_card_database(config_path)

    # 2. Download UniProt candidate negative sequences
    print("\n--- Step 2: Downloading UniProt Negative Candidates ---")
    uniprot_meta = download_uniprot_negative_candidates(max_sequences=12000, config_path=config_path)

    # 3. Download AMRFinderPlus external catalog
    print("\n--- Step 3: Downloading AMRFinderPlus External Catalog ---")
    amrfinder_meta = download_amrfinder_catalog(config_path)

    inventory = {
        "pipeline_run_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "card": card_meta,
        "uniprot": uniprot_meta,
        "amrfinder": amrfinder_meta,
    }

    # Save summary report
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = processed_dir / "dataset_inventory.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DATASET ACQUISITION COMPLETE. Inventory saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_data_pipeline()
