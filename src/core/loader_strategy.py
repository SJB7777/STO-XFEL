import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from cuptlib_config.palxfel import Hertz
from logger import AppLogger
from config import load_config, ExperimentConfiguration

import numpy.typing as npt
from typing import Union

class HDF5LoaderInterface(ABC):
    @abstractmethod
    def __init__(self, file: str) -> None:
        pass
    
    @abstractmethod
    def get_data(self) -> dict[str, npt.NDArray]:
        pass

class HDF5FileLoader(HDF5LoaderInterface):

    def __init__(self, file: str):
        """
        Initializes the HDF5FileLoader by loading metadata, images, and qbpm data from the given file.

        Parameters:
        - file (str): Path to the HDF5 file.
        """
        if not os.path.isfile(file):
            raise FileNotFoundError(f"No such file: {file}")
        
        self.file: str = file
        self.logger: AppLogger = AppLogger("MainProcessor")
        self.config: ExperimentConfiguration = load_config()
        
        metadata: pd.DataFrame = pd.read_hdf(self.file, key='metadata')
        merged_df: pd.DataFrame = self.get_merged_df(metadata)
        
        self.images: npt.NDArray[np.float32] = np.maximum(0, np.stack(merged_df['image'].values))  # Fill Negative Values to Zero
        self.qbpm: npt.NDArray[np.float32] = np.stack(merged_df['qbpm'].values)
        self.pump_status: npt.NDArray[np.bool_] = self.get_pump_mask(merged_df)
        self.delay: Union[np.float32, float] = self.get_delay(metadata)
        
        # roi_coord = np.array(self.metadata[f'detector_{self.config.param.hutch}_{self.config.param.detector}_parameters.ROI'].iloc[0][0])
        # self.roi_rect = np.array([roi_coord[self.config.param.x1], roi_coord[self.config.param.x2], roi_coord[self.config.param.y1], roi_coord[self.config.param.y2]], dtype=np.int_)

    def get_merged_df(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Merges image and qbpm data with metadata based on timestamps.

        Parameters:
        - metadata (pd.DataFrame): Metadata DataFrame.

        Returns:
        - pd.DataFrame: Merged DataFrame containing metadata, images, and qbpm data.
        """
        with h5py.File(self.file, "r") as hf:
            if "detector" not in hf:
                raise KeyError(f"Key 'detector' not found in the HDF5 file")
            
            images = np.asarray(hf[f'detector/{self.config.param.hutch}/{self.config.param.detector}/image/block0_values'], dtype=np.float32)
            images_ts = np.asarray(hf[f'detector/{self.config.param.hutch}/{self.config.param.detector}/image/block0_items'], dtype=np.int64)
            qbpm = hf[f'qbpm/{self.config.param.hutch}/qbpm1']
            qbpm_ts = qbpm[f'waveforms.ch1/axis1'][()]
            qbpm_sum = np.sum([qbpm[f'waveforms.ch{i + 1}/block0_values'] for i in range(4)], axis=0, dtype=np.float32).sum(axis=1)
        
        image_df = pd.DataFrame(
            {
                "timestamp": images_ts,
                "image": list(images)
            }
        ).set_index('timestamp')

        qbpm_df = pd.DataFrame(
            {
                "timestamp": qbpm_ts,
                "qbpm": list(qbpm_sum)
            }
        ).set_index('timestamp')

        merged_df = pd.merge(image_df, qbpm_df, left_index=True, right_index=True, how='inner')
        return pd.merge(metadata, merged_df, left_index=True, right_index=True, how='inner')
    
    def get_delay(self, metadata: pd.DataFrame) -> Union[np.float32, float]:
        """
        Retrieves the delay value from the metadata.

        Parameters:
        - metadata (pd.DataFrame): Metadata DataFrame.

        Returns:
        - Union[np.float32, float]: Delay value or NaN if not found.
        """
        if "th_value" in metadata:
            return np.asarray(metadata['th_value'], dtype=np.float32)[0]
        if "delay_value" in metadata:
            return np.asarray(metadata['delay_value'], dtype=np.float32)[0]
        else:
            self.logger.warning("'th_value' and 'delay_value' are not excisting in metadata. return NaN")
            return np.nan
    
    def get_pump_mask(self, merged_df: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """
        Generates a pump status mask based on the configuration settings.

        Parameters:
        - merged_df (pd.DataFrame): Merged DataFrame.

        Returns:
        - npt.NDArray[np.bool_]: Pump status mask.
        """
        # FIXME: config.param.pump_setting should be Hertz(Enum) but it is str now.
        # if config.param.pump_setting is Hertz.ZERO:
        if self.config.param.pump_setting == str(Hertz.ZERO):
            return np.zeros(merged_df.shape[0], dtype=np.bool_)
        return merged_df[f'timestamp_info.RATE_{self.config.param.xray}_{self.config.param.pump_setting}'].astype(bool)

    def get_data(self) -> dict[str, npt.NDArray]:
        """
        Retrieves data based on pump status.

        Returns:
        - dict[str, npt.NDArray]: Dictionary containing images and qbpm data for both pump-on and pump-off states.
        """
        data = {}
        
        pon_images = self.images[self.pump_status]
        pon_qbpm = self.qbpm[self.pump_status]
        poff_images = self.images[~self.pump_status]
        poff_qbpm = self.qbpm[~self.pump_status]
        
        if pon_images.size > 0:
            data["pon"] = pon_images
            data["pon_qbpm"] = pon_qbpm
        if poff_images.size > 0:
            data["poff"] = poff_images
            data["poff_qbpm"] = poff_qbpm

        return data

if __name__ == "__main__":
    file: str = "D:\\dev\\xfel_sample_data\\run=001\scan=001\p0110.h5"
    loader = HDF5FileLoader(file)
    data = loader.get_data()
    print(loader.delay)