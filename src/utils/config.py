"""
Configuration Loader Module
Provides unified access to YAML configuration settings.
"""

from pathlib import Path
from typing import Any, Dict
import sys
import yaml

# Auto-add project root directory to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from a YAML file.

    Args:
        config_path (str): Relative or absolute path to the configuration YAML file.

    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_absolute():
        # Relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / config_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
