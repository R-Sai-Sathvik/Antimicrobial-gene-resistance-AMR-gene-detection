"""Unit tests for configuration loading."""

from pathlib import Path
from src.utils.config import load_config


def test_load_config():
    cfg = load_config("configs/config.yaml")
    assert "project" in cfg
    assert cfg["project"]["name"] == "amr-gene-detection"
    assert "preprocessing" in cfg
    assert cfg["preprocessing"]["min_length"] == 50
    assert cfg["preprocessing"]["max_length"] == 2500
    assert "features" in cfg
    assert cfg["features"]["kmers"]["k_values"] == [3, 4]
