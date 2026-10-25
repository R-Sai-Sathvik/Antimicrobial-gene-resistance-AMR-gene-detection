"""
Advanced Protein Language Model (ESM-2) Classifier
Fine-tunes and extracts sequence embeddings using ESM-2 (35M parameters) for AMR gene detection.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from src.utils.config import load_config
from src.models.baseline_models import compute_comprehensive_metrics


class ProteinSequenceDataset(Dataset):
    """Dataset wrapper for amino acid sequences."""

    def __init__(self, sequences: List[str], labels: List[int], tokenizer, max_len: int = 1024):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
            "raw_sequence": seq,
        }


class ESM2Classifier(nn.Module):
    """ESM-2 Mean-Pooled Sequence Classifier."""

    def __init__(self, model_name: str = "facebook/esm2_t12_35M_UR50D", num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_name)
        hidden_dim = self.esm.config.hidden_size  # 480 for 35M model

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (logits, mean_pooled_embedding).
        """
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (B, L, H)

        # Mean pooling over non-padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        logits = self.classifier(mean_pooled).squeeze(-1)
        return logits, mean_pooled


def train_and_evaluate_esm2(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Train ESM-2 classifier on Train split and evaluate on Val and Test splits."""
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    metrics_dir = Path(cfg["paths"]["results_dir"]) / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ESM-2 Model] Target device: {device}")

    # Load splits
    train_df = pd.read_csv(processed_dir / "train_split.csv")
    val_df = pd.read_csv(processed_dir / "val_split.csv")
    test_df = pd.read_csv(processed_dir / "test_split.csv")

    model_name = cfg["features"]["esm"]["model_name"]
    print(f"[ESM-2 Model] Loading pre-trained tokenizer & weights: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ProteinSequenceDataset(train_df["sequence"].tolist(), train_df["label"].tolist(), tokenizer)
    val_ds = ProteinSequenceDataset(val_df["sequence"].tolist(), val_df["label"].tolist(), tokenizer)
    test_ds = ProteinSequenceDataset(test_df["sequence"].tolist(), test_df["label"].tolist(), tokenizer)

    batch_size = cfg["features"]["esm"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = ESM2Classifier(model_name=model_name).to(device)

    # Freeze base ESM-2 layers initially, train top classifier head first for speed
    for param in model.esm.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    epochs = cfg["models"]["esm2_classifier"]["epochs"]
    print(f"[ESM-2 Model] Training classifier head for {epochs} epochs...")

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        train_loss = total_loss / len(train_ds)

        # Validation evaluation
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits, _ = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(batch["label"].numpy())

        val_preds = (np.array(val_probs) >= 0.5).astype(int)
        val_metrics = compute_comprehensive_metrics(np.array(val_targets), val_preds, np.array(val_probs))

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val F1: {val_metrics['f1_score']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), models_dir / "esm2_amr_classifier.pt")

    # Final Test Set Evaluation
    print("[ESM-2 Model] Loading best model checkpoint and evaluating on Test set...")
    model.load_state_dict(torch.load(models_dir / "esm2_amr_classifier.pt"))
    model.eval()

    test_probs, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, _ = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(batch["label"].numpy())

    test_preds = (np.array(test_probs) >= 0.5).astype(int)
    test_metrics = compute_comprehensive_metrics(np.array(test_targets), test_preds, np.array(test_probs))

    results = {"validation": val_metrics, "test": test_metrics}

    with open(metrics_dir / "esm2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[ESM-2 Model] Test Set Evaluation Results:")
    print(f"  Test F1-Score         : {test_metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC          : {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC           : {test_metrics['pr_auc']:.4f}")
    print(f"  Test Sensitivity (TPR): {test_metrics['recall_sensitivity']:.4f}")
    print(f"  Test Specificity (TNR): {test_metrics['specificity']:.4f}")

    return results


if __name__ == "__main__":
    train_and_evaluate_esm2()
