"""
This module provides specific classes for managing configuration parameters of an experiment.
It builds upon the generic configuration management provided by the base_config module.

Classes:
    - `ConfigurationParameters`: A class representing configuration parameters for an experiment.
    - `ConfigurationPaths`: A class to manage configuration paths for an experiment.
    - `ExperimentConfiguration`: A dataclass representing the complete configuration for an experiment,
    combining configuration parameters and paths.
"""
import os

from pydantic import BaseModel, model_validator, Field

from src.config.enums import Hutch, Detector, Xray, Hertz


class ExpParams(BaseModel):
    """
    A dataclass to represent configuration parameters for an experiment.

    sdd = 1.3m
    dps = 75e-06m (75 um)
    beam_energy = 9.7 keV
    wavelength [A]

    Attributes:
        hutch (Hutch): The hutch setting.
        detector (Detector): The detector setting.
        xray (Xray): The x-ray setting.
        pump_setting (Hertz): The pump setting.
        x1 (int): The x1 setting.
        x2 (int): The x2 setting.
        y1 (int): The y1 setting.
        y2 (int): The y2 setting.
    """
    hutch: Hutch = Hutch.EH1
    detector: Detector = Detector.JUNGFRAU2
    xray: Xray = Xray.HARD
    pump_setting: Hertz = Hertz.FIFTEEN
    x1: int = 0
    x2: int = 1
    y1: int = 2
    y2: int = 3
    sdd: float = 1.3
    dps: float = 7.5e-5
    beam_energy: float = 9.7
    sigma_factor: float = 1
    wavelength: float = None

    @model_validator(mode='before')
    @classmethod
    def calculate_wavelength(cls, values):
        """Calculate Wavelength"""
        beam_energy = values.get('beam_energy')
        if beam_energy is not None:
            values['wavelength'] = 12.398419843320025 / beam_energy
        return values


class ExpPaths(BaseModel):
    """
    A class to manage configuration paths for an experiment.

    Attributes:
        load_dir (str): The load directory path.
        anaylsis_dir (str): The save directory path.
    """
    load_dir: str = ""
    analysis_dir: str = ""

    mat_dir: str = "mat_files"
    processed_dir: str = "processed_data"
    output_dir: str = "output_data"

    @model_validator(mode='before')
    @classmethod
    def join_paths(cls, values):
        """join paths"""
        analysis_dir = values.get('analysis_dir')
        if analysis_dir is not None:
            values['mat_dir'] = os.path.join(analysis_dir, values['mat_dir'])
            values['processed_dir'] = os.path.join(analysis_dir, values['processed_dir'])
            values['output_dir'] = os.path.join(analysis_dir, values['output_dir'])
        return values


class ExpConfig(BaseModel):
    """
    A dataclass to represent the complete configuration for an experiment.

    Attributes:
        runs (list[int]): The run numbers.
        param (ConfigurationParameters): The configuration parameters.
        path (ConfigurationPaths): The configuration paths.
    """
    runs: list[int] = Field(default_factory=list)
    param: ExpParams = ExpParams()
    path: ExpPaths = ExpPaths()


if __name__ == "__main__":
    config_dict = {
        "runs": ["1", "2", "3"],
        'path': {
            'load_dir': 'your/path',
            'anaylsis_dir': 'your/path',
            'output_dir': 'Image',
            'mat_dir': 'mat_files',
            'processed_dir': 'npz_files',
        },
        'param': {
            'xray': 'HX',
            'detector': 'jungfrau2',
            'pump_setting': '15HZ',
            'hutch': 'eh2',
            'sdd': 1.3,
            'dps': 7.5e-05,
            'beam_energy': 9.7,
            'x1': 0,
            'x2': 1,
            'y1': 2,
            'y2': 3
        }
    }

    config = ExpConfig(**config_dict)
    print(config.runs)

    print(config.path.load_dir)
    print(config.path.analysis_dir)
    print(config.path.param_dir)

    params = ExpParams(beam_energy=9.7)
    print(params.wavelength)
