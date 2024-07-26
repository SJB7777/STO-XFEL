import os

import numpy as np
from cuptlib_config.palxfel import load_palxfel_config
from scipy.io import savemat
import tifffile
from tqdm import tqdm

from delay.delay_scan import ReadDelayH5
from utils.file_util import get_run_scan_directory, get_folder_list, get_file_list
from logger import Logger

from typing import Callable, Optional

Images = np.ndarray
Qbpm = np.ndarray
Preprocess = Callable[[Images, Qbpm], tuple[Images, Qbpm]]

class DelayProcessor:
    def __init__(self, preprocessing_functions: Optional[list[Preprocess]] = None, logger: Optional[Logger] = None):
        if preprocessing_functions is None:
            preprocessing_functions = []
        self.preprocessing_functions: list[Preprocess] = preprocessing_functions
        
        self.images_dict: dict[str, np.ndarray] = {}
        if logger is None:
            self.logger = Logger("RockingProcessor")
        else:
            self.logger = logger
        
        config = load_palxfel_config("config.ini")
        self.logger.add_metadata(config.to_config_dict())


    def scan(self, run_num: int):
        self.logger.info(f"Starting scan for run number: {run_num}")
        config = load_palxfel_config("config.ini")
        root_dir = config.path.load_dir
        run_dir = get_run_scan_directory(root_dir, run_num)

        scan_folders = get_folder_list(run_dir)
        for scan_folder in scan_folders:
            scan_num = scan_folder.split("=")[1]
            scan_dir = os.path.join(run_dir, scan_folder)
            self.logger.info(f"Processing scan folder: {scan_folder}")
            images = self._single_scan(scan_dir)
            
            scan_num = int(scan_folder.split("=")[1])
            file_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"  # example: run=001_scan=001
            self.images_dict[file_name] = images
            self.logger.info(f"Completed processing for {file_name}")

    def _single_scan(self, scan_dir: str):
        self.logger.info(f"Starting single scan for directory: {scan_dir}")
        stacked_pon_images = []
        stacked_poff_images = []
        
        hdf5_files = get_file_list(scan_dir)
        pbar = tqdm(enumerate(hdf5_files), total=len(hdf5_files))
        for i, hdf5_file in pbar:
            hdf5_dir = os.path.join(scan_dir, hdf5_file)
            
            # TEMP
            rr = ReadDelayH5(hdf5_dir)
            # try:
            #     rr = ReadRockingH5(hdf5_dir)
            # except Exception as e:
            #     self.logger.error(f"Failed to load frame {i}: {type(e)}: {str(e)}")
            #     print(f"Failed to load frame {i}: {type(e)}: {str(e)}")

            #     import traceback
            #     traceback.print_exc()
            #     continue
            
            images, qbpm, pump_status = rr.images, rr.qbpm_sum, rr.pump_status

            for preprocessing_function in self.preprocessing_functions:
                images, qbpm = preprocessing_function(images, qbpm)

            pon_images, poff_images = images[pump_status], images[~pump_status]
            pon_image = pon_images.mean(axis=0)
            poff_image = poff_images.mean(axis=0)
            
            stacked_pon_images.append(pon_image)
            stacked_poff_images.append(poff_image)
        
        self.logger.info(f"Completed single scan for directory: {scan_dir}")
        
        return np.stack(stacked_pon_images), np.stack(stacked_poff_images)

    def save_as_mat(self, comment = ""):
        if not self.images_dict:
            self.logger.error("Nothing to save")
            raise Exception("Nothing to save")
        
        config = load_palxfel_config("config.ini")
        mat_dir = config.path.mat_dir

        for file_name, images in self.images_dict.items():
            mat_file = os.path.join(mat_dir, file_name + comment + ".mat")
            
            mat_format_images = images.swapaxes(0, 2)
            # TEMP
            mat_format_images = mat_format_images.swapaxes(0, 1)

            savemat(mat_file, {"data" : mat_format_images})
            self.logger.info(f"Saved MAT file Shape: {mat_format_images.shape}")
            self.logger.info(f"Saved MAT file Dtype: {mat_format_images.dtype}")
            self.logger.info(f"Saved MAT file: {mat_file}")

    def save_as_npz(self):
        if not self.images_dict:
            self.logger.error("Nothing to save")
            raise Exception("Nothing to save")
        
        config = load_palxfel_config("config.ini")
        mat_dir = config.path.mat_dir
        
        for file_name, images_dict in self.images_dict.items():
            npz_file = os.path.join(mat_dir, file_name + ".npz")
            pon_images = images_dict["pon"]
            poff_images = images_dict["poff"]
            np.savez(npz_file, pon=pon_images, poff=poff_images)
            self.logger.info(f"Saved NPZ file: {npz_file}")

    def save_as_tif(self):
        if not self.images_dict:
            self.logger.error("Nothing to save")
            raise Exception("Nothing to save")
        
        config = load_palxfel_config("config.ini")
        mat_dir = config.path.mat_dir
        
        for file_name, images in self.images_dict.items():
            tif_file = os.path.join(mat_dir, file_name + ".tif")

            tifffile.imwrite(tif_file, images)
            self.logger.info(f"Saved TIF file: {tif_file}")


if __name__ == "__main__":
    from functools import partial
    from preprocess.preprocess import nomalize_by_qbpm, filter_images_qbpm_by_linear_model, subtract_dark
    from gui.gui import find_outliers_run_scan_gui

    import setting
    setting.save()

    logger = Logger("RockingProcessor")
    run_nums = [113, 114, 115, 133, 134, 135, 156, 162, 171, 176, 177]
    logger.info(f"run: {run_nums}")

    sigma = find_outliers_run_scan_gui(run_nums[0], 1)

    remove_outlier: Preprocess = partial(filter_images_qbpm_by_linear_model, sigma=sigma)
    sub_dark: Preprocess = lambda images, qbpm : (subtract_dark(images), qbpm)
    divide_by_qbpm: Preprocess = lambda images, qbpm : (nomalize_by_qbpm(images, qbpm), qbpm)
    
    preprocessing_functions: list[Preprocess] = [
        sub_dark,
        divide_by_qbpm,
        remove_outlier,
        ]
    
    logger.info(f"preprocessing: subtract dark")
    logger.info(f"preprocessing: divide by qbpm")
    logger.info(f"preprocessing: remove outlier sigma={sigma}")
    rocking = DelayProcessor(preprocessing_functions, logger)
    
    for run_num in run_nums:
        rocking.scan(run_num)
    
    rocking.save_as_npz()
    
    logger.info("Processing is over")