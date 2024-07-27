import os
from abc import ABC, abstractmethod

from numpy.typing import NDArray
import numpy as np
from scipy.io import savemat
import tifffile
from cuptlib_config.palxfel import load_palxfel_config

class SaverStrategy(ABC):
    @abstractmethod
    def save(self, file_base_name: str, images: NDArray, comment: str=""):
        pass
    
    @property
    @abstractmethod
    def file(self) -> str:
        pass

class MatSaverStrategy(SaverStrategy):
    def save(self, file_base_name: str, images: NDArray, comment: str=""):
        config = load_palxfel_config("config.ini")
        mat_dir = config.path.mat_dir
        mat_file = os.path.join(mat_dir, file_base_name + comment + ".mat")
        
        mat_format_images = images.swapaxes(0, 2)
        mat_format_images = mat_format_images.swapaxes(0, 1)

        savemat(mat_file, {"data": mat_format_images})
        self._file_name = mat_file
    
    @property
    def file(self) -> str:
        return self._file_name

class NpzSaverStrategy(SaverStrategy):
    def save(self, file_base_name: str, images: NDArray, comment: str=""):
        config = load_palxfel_config("config.ini")
        npz_dir = config.path.npz_dir
        npz_file = os.path.join(npz_dir, file_base_name + comment + ".npz")
        
        np.savez(npz_file, data=images)
        self._file_name = npz_file
        
    @property
    def file(self) -> str:
        return self._file_name
    
class TifSaverStrategy(SaverStrategy):
    def save(self, file_base_name: str, images: NDArray, comment: str=""):
        config = load_palxfel_config("config.ini")
        tif_dir = config.path.tif_dir
        tif_file = os.path.join(tif_dir, file_base_name + comment + ".tif")

        tifffile.imwrite(tif_file, images.astype(np.float32))
        self._file_name = tif_file
        
    @property
    def file(self) -> str:
        return self._file_name

class SaverFactory:
    @staticmethod
    def get_saver(file_type) -> SaverStrategy:
        if file_type == 'mat':
            return MatSaverStrategy()
        elif file_type == 'npz':
            return NpzSaverStrategy()
        elif file_type == 'tif':
            return TifSaverStrategy()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
