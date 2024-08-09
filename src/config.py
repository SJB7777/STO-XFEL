import os
import configparser

from cuptlib_config.palxfel import load_palxfel_config, save_palxfel_dict, ExperimentConfiguration


config: configparser.ConfigParser = configparser.ConfigParser()

config_file: str = "config\\config.ini"
config.read(config_file)
config_dir: str = config["config"]["config_dir"]


def load_config() -> ExperimentConfiguration:
    """load config file and return config object"""
    return load_palxfel_config(config_dir)


def save_config(config_dict: dict) -> None:
    """get config dict and save to file"""
    save_palxfel_dict(config_dict, config_dir)


if __name__ == "__main__":
    from cuptlib_config.palxfel.enums import Hertz, Hutch, Detector, Xray

    # sdd = 1.3 # m
    # dps = 75e-06 # m (73 um)
    # beam_energy = 9.7 # keV
    # wavelength [A]

    config_dict = {
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
            "hutch": Hutch.EH1,
            # Detector
            "detector": Detector.JUNGFRAU2,
            # Xray used in experiment.
            "xray": Xray.HARD,
            # Rate of laser.
            "pump_setting": Hertz.FIFTEEN,
            # Index of roi coordinate inside h5 file.
            "x1": 0, "x2": 1, "y1": 2, "y2": 3,
            # Metric of SDD and DPS is meters.
            "sdd": 1.3,
            "dps": 7.5e-5,  # Detector Pixel Size
            "beam_energy": 9.7,
        }
    }

    save_config(config_dict)
