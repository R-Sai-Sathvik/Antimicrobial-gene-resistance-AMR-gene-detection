"""
Physicochemical Sequence Property Extractor
Computes global biophysical properties (GRAVY, Isoelectric Point, Molecular Weight, Charge).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
from typing import List, Dict, Any
import pickle
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config


VALID_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def compute_sequence_properties(seq: str) -> List[float]:
    """
    Calculate 6 biophysical properties for an amino acid sequence string.

    Returns:
        List[float]: [Length, Isoelectric_Point, Molecular_Weight, GRAVY, Aromaticity, Charge_pH7]
    """
    seq_clean = "".join([c for c in seq.strip().upper() if c in VALID_AA_SET])
    if len(seq_clean) < 10:
        return [len(seq), 6.0, 1000.0, 0.0, 0.0, 0.0]

    try:
        analysis = ProteinAnalysis(seq_clean)
        length = float(len(seq_clean))
        pi = float(analysis.isoelectric_point())
        mw = float(analysis.molecular_weight())
        gravy = float(analysis.gravy())
        aromaticity = float(analysis.aromaticity())
        charge = float(analysis.charge_at_pH(7.0))
        return [length, pi, mw, gravy, aromaticity, charge]
    except Exception:
        return [float(len(seq_clean)), 6.0, 10000.0, 0.0, 0.0, 0.0]


class PhysicochemicalExtractor:
    """Extracts and standardizes biophysical features."""

    FEATURE_NAMES = ["Length", "Isoelectric_Point", "Molecular_Weight", "GRAVY", "Aromaticity", "Charge_pH7"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, sequences: List[str]) -> "PhysicochemicalExtractor":
        """Fit scaler ONLY on training sequence properties."""
        print(f"[Biophysical] Computing properties and fitting scaler on {len(sequences)} training sequences...")
        raw_feats = np.array([compute_sequence_properties(s) for s in sequences])
        self.scaler.fit(raw_feats)
        self.is_fitted = True
        return self

    def transform(self, sequences: List[str]) -> np.ndarray:
        """Transform sequences into scaled biophysical feature matrix."""
        if not self.is_fitted:
            raise ValueError("PhysicochemicalExtractor must be fitted on training data first!")
        raw_feats = np.array([compute_sequence_properties(s) for s in sequences])
        return self.scaler.transform(raw_feats)

    def save(self, filepath: Path) -> None:
        """Save fitted scaler to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, filepath: Path) -> "PhysicochemicalExtractor":
        """Load fitted scaler from disk."""
        with open(filepath, "rb") as f:
            scaler = pickle.load(f)
        obj = cls()
        obj.scaler = scaler
        obj.is_fitted = True
        return obj


def extract_and_save_biophysical_features(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load train/val/test splits, extract scaled biophysical features, and save numpy arrays."""
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])

    train_df = pd.read_csv(processed_dir / "train_split.csv")
    val_df = pd.read_csv(processed_dir / "val_split.csv")
    test_df = pd.read_csv(processed_dir / "test_split.csv")

    extractor = PhysicochemicalExtractor()
    extractor.fit(train_df["sequence"].tolist())

    X_train = extractor.transform(train_df["sequence"].tolist())
    X_val = extractor.transform(val_df["sequence"].tolist())
    X_test = extractor.transform(test_df["sequence"].tolist())

    np.savez_compressed(
        processed_dir / "biophysical_features.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
    )
    extractor.save(processed_dir / "biophysical_scaler.pkl")

    print(f"[Biophysical] Biophysical features saved. Feature matrix shape: {X_train.shape}")
    return {"feature_names": PhysicochemicalExtractor.FEATURE_NAMES, "shape": list(X_train.shape)}


if __name__ == "__main__":
    extract_and_save_biophysical_features()
