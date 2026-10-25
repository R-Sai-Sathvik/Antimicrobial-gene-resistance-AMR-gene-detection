import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import load_config
from src.models.esm2_model import ESM2Classifier, ProteinSequenceDataset

def main():
    cfg = load_config("configs/config.yaml")
    test_df = pd.read_csv("data/processed/test_split.csv")
    print(f"Loaded test_split.csv with {len(test_df)} sequences.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model_name = cfg["features"]["esm"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = ESM2Classifier(model_name=model_name).to(device)

    ckpt_path = "models/esm2_amr_classifier.pt"
    esm_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    esm_model.eval()
    print("ESM-2 model checkpoint loaded.", flush=True)

    test_ds = ProteinSequenceDataset(test_df["sequence"].tolist(), test_df["label"].tolist(), tokenizer)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    probs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, _ = esm_model(input_ids, attention_mask)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p)
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(loader)} done...", flush=True)

    probs = np.array(probs, dtype=np.float32)
    os.makedirs("results/metrics", exist_ok=True)
    np.save("results/metrics/esm2_test_probs.npy", probs)
    print(f"Saved {len(probs)} ESM-2 test probabilities to results/metrics/esm2_test_probs.npy", flush=True)

if __name__ == "__main__":
    main()
