from pathlib import Path
from functools import lru_cache

import yaml

from src.config.config_definitions import ExpConfig


@lru_cache(maxsize=1)
def load_config() -> ExpConfig:
    """load config file and return config object"""
    config_file = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    with open(config_file, 'r', encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return ExpConfig(**config_dict)


def save_config(config_dict: dict) -> None:
    """get config dict and save to file"""
    config_file = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    with open(config_file, 'w', encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
