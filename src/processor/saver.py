import os
from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np
from scipy.io import savemat
import tifffile

from src.config.config import load_config


class SaverStrategy(ABC):
    """Save data_dict to a file."""
    @abstractmethod
    def save(self, file_base_name: str, data_dict: dict[str, npt.NDArray], comment: str = ""):
        pass

    @property
    @abstractmethod
    def file(self) -> str:
        pass

    @property
    @abstractmethod
    def file_type(self) -> str:
        pass


class MatSaverStrategy(SaverStrategy):
    def save(self, file_base_name: str, data_dict: dict[str, npt.NDArray], comment: str = ""):
        comment = "_" + comment if comment else ""
        config = load_config()
        mat_dir = config.path.mat_dir
        os.makedirs(mat_dir, exist_ok=True)
        for key, val in data_dict.items():
            if val.ndim == 3:
                mat_format_images = val.swapaxes(0, 2).swapaxes(0, 1)  # TEMP
                mat_file = os.path.join(mat_dir, f"{file_base_name}_{key}{comment}.mat")
                savemat(mat_file, {"data": mat_format_images})
        self._file_name = mat_file

    @property
    def file(self) -> str:
        return self._file_name

    @property
    def file_type(self) -> str:
        return "mat"


class NpzSaverStrategy(SaverStrategy):
    def save(self, file_base_name: str, data_dict: dict[str, npt.NDArray], comment: str = ""):
        comment = "_" + comment if comment else ""
        config = load_config()
        npz_dir = config.path.npz_dir
        os.makedirs(npz_dir, exist_ok=True)
        npz_file = os.path.join(npz_dir, file_base_name + comment + ".npz")
        np.savez(npz_file, **data_dict)
        self._file_name = npz_file

    @property
    def file(self) -> str:
        return self._file_name

    @property
    def file_type(self) -> str:
        return "npz"


class TifSaverStrategy(SaverStrategy):
    def save(self, file_base_name: str, data_dict: dict[str, npt.NDArray], comment: str = ""):
        comment = "_" + comment if comment else ""
        config = load_config()
        tif_dir = config.path.tif_dir
        os.makedirs(tif_dir, exist_ok=True)
        for key, val in data_dict.items():
            if val.ndim == 3:
                tif_file = os.path.join(tif_dir, f"{file_base_name}_{key}{comment}.tif")
                tifffile.imwrite(tif_file, val.astype(np.float32))
        self._file_name = tif_file

    @property
    def file(self) -> str:
        return self._file_name

    @property
    def file_type(self) -> str:
        return "tif"


class SaverFactory:
    """Get SaverStrategy"""

    @staticmethod
    def get_saver(file_type) -> SaverStrategy:
        """return SaverStrategy"""
        if file_type == 'mat':
            return MatSaverStrategy()
        if file_type == 'npz':
            return NpzSaverStrategy()
        if file_type == 'tif':
            return TifSaverStrategy()
        raise ValueError(f"Unsupported file type: {file_type}")
