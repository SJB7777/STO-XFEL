import os

import h5py
import hdf5plugin
import numpy as np
import pandas as pd

from cuptlib_config.palxfel import load_palxfel_config

class ReadDelayH5:
    def __init__(self, file_path: str):
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: {file_path}")
        
        config = load_palxfel_config()
        self.file_path = file_path
        self.metadata = pd.read_hdf(self.file_path, 'metadata')

        self.delay = np.array(self.metadata['delay_value'])[0]
        self.pump_status = self.metadata[f'timestamp_info.RATE_{config.param.xray}_{config.param.pump_setting}']
        
        with h5py.File(self.file_path, 'r') as file:
            self.images = np.asarray(file[f'detector/{config.param.hutch}/{config.param.detector}/image/block0_values'])
            self.images_ts = np.asarray(file[f'detector/{config.param.hutch}/{config.param.detector}/image/block0_items'])
            
            qbpm = file[f'qbpm/{config.param.hutch}/qbpm1']
            self.qbpm_ts = qbpm[f'waveforms.ch1/axis1'][()]
            self.qbpm_sum = np.sum([qbpm[f'waveforms.ch{i + 1}/block0_values'] for i in range(4)], axis=0).sum(axis=1)

        self.merged_df = self.get_merged_df()
        
        self.images = self.merged_df['image']
        self.qbpm_sum = self.merged_df['qbpm']
        self.pump_status = self.merged_df[f'timestamp_info.RATE_{config.param.xray}_{config.param.pump_setting}'].astype(bool)
        
        self.images = np.stack(self.images.values)
        self.qbpm_sum = np.stack(self.qbpm_sum.values)
        self.images = np.maximum(0, self.images)
        
        # Divide images by qbpm image.
        # self.images = self.images / self.qbpm_sum[:, np.newaxis, np.newaxis]
        
        # self.pon_images = self.images[self.pump_status]
        # self.poff_images = self.images[~self.pump_status]

        # self.pon_qbpm = self.qbpm_sum[self.pump_status]
        # self.poff_qbpm = self.qbpm_sum[~self.pump_status]

        # roi_coord = np.array(self.metadata[f'detector_{self.hutch}_{self.detector}_parameters.ROI'].iloc[0][0])
        # self.roi_rect = np.array([roi_coord[self.x1], roi_coord[self.x2], roi_coord[self.y1], roi_coord[self.y2]], dtype=np.dtype(int))
    def get_merged_df(self) -> pd.DataFrame:
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