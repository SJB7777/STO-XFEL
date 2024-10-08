import os
from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np
from scipy.io import savemat

from src.config.config import load_config
from src.utils.file_util import create_run_scan_directory


def get_file_base_name(run_n: int, scan_n: int) -> str:
    """Return formated file name"""
    return f'run={run_n:04}_scan={scan_n:04}'


class SaverStrategy(ABC):
    """Save data_dict to a file."""
    @abstractmethod
    def save(self, run_n: int, scan_n: int, data_dict: dict[str, npt.NDArray], comment: str = ""):
        pass

    @property
    @abstractmethod
    def file(self) -> str:
        """Return File Name"""

    @property
    @abstractmethod
    def file_type(self) -> str:
        """Return File Type"""


class MatSaverStrategy(SaverStrategy):

    def __init__(self):
        self._file: str = None

    def save(self, run_n: int, scan_n: int, data_dict: dict[str, npt.NDArray], comment: str = ""):
        comment = "_" + comment if comment else ""
        config = load_config()
        mat_dir = config.path.mat_dir
        os.makedirs(mat_dir, exist_ok=True)
        for key, val in data_dict.items():
            if val.ndim == 3:
                mat_format_images = val.swapaxes(0, 2).swapaxes(0, 1)  # TEMP
                file_base_name = get_file_base_name(run_n, scan_n)
                mat_file = os.path.join(mat_dir, f"{file_base_name}_{key}{comment}.mat")
                savemat(mat_file, {"data": mat_format_images})
        self._file = mat_file

    @property
    def file(self) -> str:
        return self._file

    @property
    def file_type(self) -> str:
        return "mat"


class NpzSaverStrategy(SaverStrategy):

    def __init__(self):
        self._file: str = None

    def save(self, run_n: int, scan_n: int, data_dict: dict[str, npt.NDArray], comment: str = ""):
        comment = "_" + comment if comment else ""
        config = load_config()
        processed_dir = config.path.processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        file_base_name = get_file_base_name(run_n, scan_n)
        npz_dir = create_run_scan_directory(processed_dir, run_n, scan_n)
        npz_file = os.path.join(npz_dir, file_base_name + comment + ".npz")
        np.savez(npz_file, **data_dict)
        self._file = npz_file

    @property
    def file(self) -> str:
        return self._file

    @property
    def file_type(self) -> str:
        return "npz"


def get_saver_strategy(file_type: str) -> SaverStrategy:
    """Get SaverStrategy by file type."""
    strategies = {
        'mat': MatSaverStrategy,
        'npz': NpzSaverStrategy
    }

    strategy_class = strategies.get(file_type)
    if strategy_class is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return strategy_class()
