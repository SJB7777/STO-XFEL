import os

import numpy as np
from cuptlib_config.palxfel import load_palxfel_config
from scipy.io import savemat
import tifffile

from rocking.rocking_scan import ReadRockingH5
from utils.file_util import get_run_scan_directory, get_folder_list, get_file_list
from logger import Logger

from typing import Callable


Preprocess = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

class RockingProcessor:
    
    def __init__(self, preprocessing_functions: list[Preprocess] | None = None):
        if preprocessing_functions is None:
            preprocessing_functions = []
        self.preprocessing_functions: list[Preprocess] = preprocessing_functions
        
        self.images_dict: dict[str, np.ndarray] = {}
        self.logger = Logger("RockingProcessor")

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
            file_name: str = f"run={run_num:0>3}_scan={scan_num:0>3}"  # example: run=001_scan=001
            self.images_dict[file_name] = images
            self.logger.info(f"Completed processing for {file_name}")

    def _single_scan(self, scan_dir: str):
        self.logger.info(f"Starting single scan for directory: {scan_dir}")
        stacked_images = []
        
        hdf5_files = get_file_list(scan_dir)
        for i, hdf5_file in enumerate(hdf5_files):
            hdf5_dir = os.path.join(scan_dir, hdf5_file)
            
            try:
                rr = ReadRockingH5(hdf5_dir)
            except Exception as e:
                self.logger.error(f"Failed to load frame {i}: {type(e)}: {str(e)}")
                print(f"Failed to load frame {i}: {type(e)}: {str(e)}")
                continue
            
            images, qbpm = rr.images, rr.qbpm_sum
            for preprocessing_function in self.preprocessing_functions:
                images, qbpm = preprocessing_function(images, qbpm)
            
            stacked_images.append(images)
        
        self.logger.info(f"Completed single scan for directory: {scan_dir}")
        return np.stack(stacked_images)
    
    def save_as_mat(self):
        if not self.images_dict:
            self.logger.error("Nothing to save")
            raise Exception("Nothing to save")
        
        config = load_palxfel_config("config.ini")
        mat_dir = config.path.mat_dir
        print(mat_dir)
        for file_name, images in self.images_dict.items():
            mat_file = os.path.join(mat_dir, file_name + ".mat")
            
            mat_format_images = images.swapaxes(0, 2)
            savemat(mat_file, {"data" : mat_format_images})
            self.logger.info(f"Saved MAT file: {mat_file}")

    def save_as_npz(self):
        if not self.images_dict:
            self.logger.error("Nothing to save")
            raise Exception("Nothing to save")
        
        config = load_palxfel_config("config.ini")
        mat_dir = config.path.mat_dir
        
        for file_name, images in self.images_dict.items():
            npz_file = os.path.join(mat_dir, file_name + ".npz")
            
            np.savez(npz_file, data=images)
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
    from preprocess.preprocess import nomalize_by_qbpm, filter_images_qbpm_by_linear_model
    
    import setting
    setting.save()

    logger = Logger("RockingProcessor")
    
    run_nums = [1]
    logger.info(f"run: {run_nums}")
    
    remove_outlier = partial(filter_images_qbpm_by_linear_model, sigma=3)
    divide_by_qbpm = lambda images, qbpm : (nomalize_by_qbpm(images, qbpm), qbpm)
    
    preprocessing_functions: list[Preprocess] = [remove_outlier, divide_by_qbpm]
    rocking = RockingProcessor(preprocessing_functions)
    
    for run_num in run_nums:
        rocking.scan(run_num)
    
    rocking.save_as_mat()
    
    logger.info("Processing is over")