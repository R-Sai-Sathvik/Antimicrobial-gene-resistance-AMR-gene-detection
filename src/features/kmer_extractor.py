"""
Amino Acid k-mer Feature Extractor
Extracts normalized 3-mer and 4-mer frequency representations with TF-IDF weighting.
Ensures vectorizer fitting is strictly restricted to training data to eliminate data leakage.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
from typing import List, Tuple, Dict, Any
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.utils.config import load_config


def sequence_to_kmer_string(seq: str, k: int) -> str:
    """Convert an amino acid sequence string into space-separated k-mer tokens."""
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    return " ".join(kmers)


class KmerFeatureExtractor:
    """Extracts k-mer TF-IDF features from protein sequences."""

    def __init__(self, k_values: List[int] = [3, 4], max_features: int = 5000):
        self.k_values = k_values
        self.max_features = max_features
        self.vectorizers: Dict[int, TfidfVectorizer] = {}
        self.is_fitted = False

    def fit(self, sequences: List[str]) -> "KmerFeatureExtractor":
        """
        Fit TF-IDF vectorizers ONLY on training sequences.
        """
        print(f"[KmerExtractor] Fitting TF-IDF vectorizers for k={self.k_values} on {len(sequences)} training sequences...")
        for k in self.k_values:
            corpus = [sequence_to_kmer_string(s, k) for s in sequences]
            vec = TfidfVectorizer(
                token_pattern=r"\S+",
                max_features=self.max_features // len(self.k_values),
                norm="l2",
                sublinear_tf=True,
            )
            vec.fit(corpus)
            self.vectorizers[k] = vec
        self.is_fitted = True
        return self

    def transform(self, sequences: List[str]) -> np.ndarray:
        """
        Transform sequence list into fixed-size TF-IDF feature matrix.
        """
        if not self.is_fitted:
            raise ValueError("KmerFeatureExtractor must be fitted on training data before transforming!")

        feature_blocks = []
        for k in self.k_values:
            corpus = [sequence_to_kmer_string(s, k) for s in sequences]
            mat = self.vectorizers[k].transform(corpus).toarray()
            feature_blocks.append(mat)

        return np.hstack(feature_blocks)

    def save(self, filepath: Path) -> None:
        """Save fitted vectorizers to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({"k_values": self.k_values, "vectorizers": self.vectorizers}, f)

    @classmethod
    def load(cls, filepath: Path) -> "KmerFeatureExtractor":
        """Load fitted vectorizers from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls(k_values=data["k_values"])
        obj.vectorizers = data["vectorizers"]
        obj.is_fitted = True
        return obj


def extract_and_save_kmer_features(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load train/val/test splits, extract k-mer features without leakage, and save numpy arrays."""
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])

    train_df = pd.read_csv(processed_dir / "train_split.csv")
    val_df = pd.read_csv(processed_dir / "val_split.csv")
    test_df = pd.read_csv(processed_dir / "test_split.csv")

    extractor = KmerFeatureExtractor(
        k_values=cfg["features"]["kmers"]["k_values"],
        max_features=4000
    )

    # FIT ONLY ON TRAIN
    extractor.fit(train_df["sequence"].tolist())

    X_train = extractor.transform(train_df["sequence"].tolist())
    X_val = extractor.transform(val_df["sequence"].tolist())
    X_test = extractor.transform(test_df["sequence"].tolist())

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    np.savez_compressed(
        processed_dir / "kmer_features.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )

    extractor.save(processed_dir / "kmer_extractor.pkl")

    stats = {
        "kmer_feature_dim": X_train.shape[1],
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
        "X_test_shape": list(X_test.shape),
    }

    print(f"[KmerExtractor] Features saved. Feature matrix shape: {X_train.shape}")
    return stats


if __name__ == "__main__":
    extract_and_save_kmer_features()
