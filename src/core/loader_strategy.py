import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from cuptlib_config.palxfel import load_palxfel_config, Hertz
from logger import AppLogger

from preprocess.image_qbpm_processors import ImagesQbpmProcessor
import numpy.typing as npt

class HDF5LoaderInterface(ABC):
    @abstractmethod
    def __init__(self, file: str) -> None:
        pass
    
    @abstractmethod
    def _get_merged_df(self) -> pd.DataFrame:
        """
        Merges image and qbpm data with metadata, removing rows with missing data.

        This method combines the image and qbpm data with metadata based on timestamp,
        removes rows with missing data, and updates the class attributes `images`, 
        `qbpm_sum`, and `pump_status` accordingly.

        Returns:
            DataFrame
        """
        pass
    
    @abstractmethod
    def get_images_dict(self) -> dict[str, npt.NDArray]:
        pass
    
    @property
    @abstractmethod
    def delay(self) -> npt.NDArray:
        pass

class HDF5FileLoader(HDF5LoaderInterface):
    """
    Initializes the RockingScan object by loading metadata, images, and qbpm data from the given file.

    Parameters:
    - file (str): Path to the HDF5 file.
    """

    def __init__(self, file: str):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"No such file: {file}")
        self.logger = AppLogger("MainProcessor")
        config = load_palxfel_config("config.ini")
        
        self.metadata = pd.read_hdf(file, 'metadata')
        
        if "th_value" in self.metadata:
            self._delay = np.asarray(self.metadata['th_value'])[0]
        elif "delay_value" in self.metadata:
            self._delay = np.asarray(self.metadata['delay_value'])[0]
        else:
            self.logger.warning("'th_value' and 'delay_value' are not excisting in metadata.")
            self._delay = np.nan
            
        with h5py.File(file) as hf:
            if "detector" not in hf:
                raise KeyError(f"Key 'detector' not found in the HDF5 file")
            self.images = np.asarray(hf[f'detector/{config.param.hutch}/{config.param.detector}/image/block0_values'])
            self.images_ts = np.asarray(hf[f'detector/{config.param.hutch}/{config.param.detector}/image/block0_items'])
            
            qbpm = hf[f'qbpm/{config.param.hutch}/qbpm1']
            self.qbpm_ts = qbpm[f'waveforms.ch1/axis1'][()]
            self.qbpm_sum = np.sum([qbpm[f'waveforms.ch{i + 1}/block0_values'] for i in range(4)], axis=0).sum(axis=1)
        
        self.merged_df = self._get_merged_df()

        self.images = self.merged_df['image']
        self.qbpm_sum = self.merged_df['qbpm']
        
        self.images = np.stack(self.images.values)
        self.qbpm_sum = np.stack(self.qbpm_sum.values)

        # FIXME: config.param.pump_setting should be Hertz(Enum) but it is str now.
        # if config.param.pump_setting is Hertz.ZERO:
        if config.param.pump_setting == str(Hertz.ZERO):
            self.pump_status = np.zeros_like(self.qbpm_sum, dtype=np.bool_)
        else:
            self.pump_status = self.merged_df[f'timestamp_info.RATE_{config.param.xray}_{config.param.pump_setting}'].astype(bool)

        # Remove below zero.
        self.images = np.maximum(0, self.images)

        self.pon_images = self.images[self.pump_status]
        self.poff_images = self.images[~self.pump_status]

        # roi_coord = np.array(self.metadata[f'detector_{config.param.hutch}_{config.param.detector}_parameters.ROI'].iloc[0][0])
        # self.roi_rect = np.array([roi_coord[self.x1], roi_coord[self.x2], roi_coord[self.y1], roi_coord[self.y2]], dtype=np.dtype(int))
        
    def _get_merged_df(self) -> pd.DataFrame:
        """
        Merges image and qbpm data with metadata, removing rows with missing data.

        This method combines the image and qbpm data with metadata based on timestamp,
        removes rows with missing data, and updates the class attributes `images`, 
        `qbpm_sum`, and `pump_status` accordingly.

        Returns:
            DataFrame
        """
        image_df = pd.DataFrame(
            {
                "timestamp": self.images_ts,
                "image": list(self.images)
            }
        ).set_index('timestamp')

        qbpm_df = pd.DataFrame(
            {
                "timestamp": self.qbpm_ts,
                "qbpm": list(self.qbpm_sum)
            }
        ).set_index('timestamp')
        
        merged_df = pd.merge(image_df, qbpm_df, left_index=True, right_index=True, how='inner')

        return pd.merge(self.metadata, merged_df, left_index=True, right_index=True, how='inner')
    
    def get_images_dict(self) -> dict[str, npt.NDArray]:
        data = {}
        
        if self.pon_images.size > 0:
            data["pon"] = self.pon_images
            data["pon_qbpm"] = self.qbpm_sum[self.pump_status]
        if self.poff_images.size > 0:
            data["poff"] = self.poff_images
            data["poff_qbpm"] = self.qbpm_sum[~self.pump_status]
            
        return data
    
    @property
    def delay(self) -> npt.NDArray:
        return self._delay