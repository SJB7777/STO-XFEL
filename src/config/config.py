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


if __name__ == "__main__":
    from src.config.enums import Hutch, Detector, Xray, Hertz

    # config = load_config()
    # print(config.runs)
    config_dict = {
        "runs": [1, 2, 3],
        "path": {
            # Mother Directory of run files.
            "load_dir": "Y:\\240608_FXS\\raw_data\\h5\\type=raw",
            "save_dir": "Y:\\240608_FXS\\raw_data\\h5\\type=raw",
            # "load_dir": "D:\\dev\\p_python\\xrd\\xfel_sample_data",
            # "save_dir": "D:\\dev\\p_python\\xrd\\xfel_sample_data",
            # relative path based on save_dir
            "image_dir": "Image",
            "param_dir": "DataParameter",
            "mat_dir": "Mat_files2",
            "npz_dir": "Npz_files",
            "tif_dir": "Tif_files"
        },
        "param": {
            # Hutch
            "hutch": Hutch.EH1.value,
            # Detector
            "detector": Detector.JUNGFRAU2.value,
            # Xray used in experiment.
            "xray": Xray.HARD.value,
            # Rate of laser.
            "pump_setting": Hertz.FIFTEEN.value,
            # Index of roi coordinate inside h5 file.
            "x1": 0, "x2": 1, "y1": 2, "y2": 3,
            # Metric of SDD and DPS is meters.
            "sdd": 1.3,
            "dps": 7.5e-5,  # Detector Pixel Size
            "beam_energy": 9.7,
        }
    }

    save_config(config_dict)
