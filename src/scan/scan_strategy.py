import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from cuptlib_config.palxfel import load_palxfel_config

from preprocess.image_qbpm_processors import ImageQbpmProcessor
import numpy.typing as npt


class ScanStrategy(ABC):
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
    def apply_preprocessing_functions(self, preprocessing_functions: list[ImageQbpmProcessor]):
        """
        Applies a list of preprocessing functions to the images and Qbpm sum.

        Parameters:
        - preprocessing_functions (List[ImageQbpmProcessor]): List of preprocessing functions to apply.
        """
        pass
    
    @abstractmethod
    def get_data(self) -> dict[str, npt.NDArray]:
        pass


class RockingScan(ScanStrategy):
    """
    Initializes the RockingScan object by loading metadata, images, and qbpm data from the given file.

    Parameters:
    - file (str): Path to the HDF5 file.
    """

    def __init__(self, file: str):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"No such file: {file}")

        config = load_palxfel_config("config.ini")

        self.metadata = pd.read_hdf(file, 'metadata')
        # self.theta = self.metadata['th_value']
        with h5py.File(file) as hf:
            if "detector" not in hf:
                raise KeyError(f"Key 'detector' not found in the HDF5 file")
            self.images = np.asarray(hf[f'detector/{config.param.hutch}/{config.param.detector}/image/block0_values'])
            self.images_ts = np.asarray(hf[f'detector/{config.param.hutch}/{config.param.detector}/image/block0_items'])
            
            # FIXME: KeyError: 'th_value'
            # self.theta = np.asarray(self.metadata['th_value'])[0]
            qbpm = hf[f'qbpm/{config.param.hutch}/qbpm1']
            self.qbpm_ts = qbpm[f'waveforms.ch1/axis1'][()]
            self.qbpm_sum = np.sum([qbpm[f'waveforms.ch{i + 1}/block0_values'] for i in range(4)], axis=0).sum(axis=1)
        
        self.merged_df = self._get_merged_df()

        self.images = self.merged_df['image']
        self.qbpm_sum = self.merged_df['qbpm']
        self.images = np.stack(self.images.values)
        self.qbpm_sum = np.stack(self.qbpm_sum.values)
        self.pump_status = self.merged_df[f'timestamp_info.RATE_{config.param.xray}_{config.param.pump_setting}'].astype(bool)
        self.pon_images = self.images[self.pump_status]
        self.poff_images = self.images[~self.pump_status]
        # roi_coord = np.array(self.metadata[f'detector_{config.param.hutch}_{config.param.detector}_parameters.ROI'].iloc[0][0])

        # Remove below zero.
        self.images = np.maximum(0, self.images)
        # roi_coord = np.array(self.metadata[f'detector_{self.hutch}_{self.detector}_parameters.ROI'].iloc[0][0])
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
    
    def apply_preprocessing_functions(self, preprocessing_functions: list[ImageQbpmProcessor]):
        
        for function in preprocessing_functions:
            self.images, self.qbpm_sum = function(self.images, self.qbpm_sum)

    def get_data(self) -> dict[str, npt.NDArray]:
        data = {}
        
        if self.pon_images.size > 0:
            data["pon"] = self.pon_images.mean(axis=0)
        if self.poff_images.size > 0:
            data["poff"] = self.poff_images.mean(axis=0)
        
        return data