import os

import numpy as np
from cuptlib_config.palxfel import load_palxfel_config
from scipy.io import savemat
import tifffile
from tqdm import tqdm

from rocking.rocking_scan import ReadRockingH5
from utils.file_util import get_run_scan_directory, get_folder_list, get_file_list
from save.saver import SaverStrategy
from logger import AppLogger

from typing import Callable, Optional
import numpy.typing as npt
Images = npt.NDArray
Qbpm = npt.NDArray
Preprocess = Callable[[Images, Qbpm], tuple[Images, Qbpm]]

class RockingProcessor:
    
    def __init__(self, preprocessing_functions: Optional[list[Preprocess]] = None, logger: Optional[AppLogger] = None):
        """
        Initializes the RockingProcessor instance.

        Parameters:
        - preprocessing_functions (list[Preprocess], optional): List of preprocessing functions to apply to images.
        - logger (Logger, optional): Logger instance for logging messages.
        """
        
        if preprocessing_functions is None:
            preprocessing_functions = []
        self.preprocessing_functions: list[Preprocess] = preprocessing_functions
        
        self.images_dict: dict[str, np.ndarray] = {}
        if logger is None:
            self.logger = AppLogger("RockingProcessor")
        else:
            self.logger = logger
        
        config = load_palxfel_config("config.ini")
        self.logger.add_metadata(config.to_config_dict())

    def scan(self, run_num: int):
        """
        Scans directories for rocking scan data and processes them.

        Parameters:
        - run_num (int): Run number to scan.
        """
        self.logger.info(f"Starting scan for run number: {run_num}")
        self.images_dict.clear()

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
            file_base_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"  # example: run=001_scan=001
            self.images_dict[file_base_name] = images
            self.logger.info(f"Completed processing for {file_base_name}")

    def _single_scan(self, scan_dir: str):
        """
        Processes a single scan directory.

        Parameters:
        - scan_dir (str): Directory path of the scan to process.

        Returns:
        - np.ndarray: Stacked images from the scan.
        """
        self.logger.info(f"Starting single scan for directory: {scan_dir}")
        stacked_images = []
        
        hdf5_files = get_file_list(scan_dir)
        pbar = tqdm(enumerate(hdf5_files), total=len(hdf5_files))
        for i, hdf5_file in pbar:
            hdf5_dir = os.path.join(scan_dir, hdf5_file)
            
            try:
                rr = ReadRockingH5(hdf5_dir)
            except KeyError as e:
                self.logger.warning(f"{e}")
                self.logger.warning(f"KeyError happened in {scan_dir}")
            # try:
            #     rr = ReadRockingH5(hdf5_dir)
            # except Exception as e:
            #     self.logger.error(f"Failed to load frame {i}: {type(e)}: {str(e)}")
            #     print(f"Failed to load frame {i}: {type(e)}: {str(e)}")

            #     import traceback
            #     traceback.print_exc()
            #     continue
            
            images, qbpm = rr.images, rr.qbpm_sum

            for preprocessing_function in self.preprocessing_functions:
                images, qbpm = preprocessing_function(images, qbpm)

            image = images.mean(axis=0)
            stacked_images.append(image)
        
        self.logger.info(f"Completed single scan for directory: {scan_dir}")
        return np.stack(stacked_images)

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

    def save(self, saver: SaverStrategy, comment: str=""):
        """
        Saves processed images using a specified saving strategy.

        Parameters:
        - saver (SaverStrategy): Saving strategy to use.
        - comment (str, optional): Comment to append to the file name.
        """
        if not self.images_dict:
            logger.error("Nothing to save")
            raise Exception("Nothing to save")
        
        for file_base_name, images in self.images_dict.items():
            saver.save(file_base_name, images, comment)
            self.logger.info(f"Images Shape: {images.shape}")
            self.logger.info(f"Images Dtype: {images.dtype}")            
            self.logger.info(f"Saved file '{saver.file}'")
        


if __name__ == "__main__":
    from functools import partial
    from preprocess.preprocess import nomalize_by_qbpm, filter_images_qbpm_by_linear_model, subtract_dark, RANSAC_regression
    from gui.preprocess_gui import find_outliers_run_scan_gui
    from save.saver import SaverFactory

    logger: AppLogger = AppLogger("RockingProcessor")
    run_nums: list[int] = [1]
    logger.info(f"run: {run_nums}")

    for run_num in run_nums:
        
        # Preprocessing functions
        # sigma = find_outliers_run_scan_gui(run_nums[0], 1)
        # remove_outlier: Preprocess = partial(filter_images_qbpm_by_linear_model, sigma=sigma)
        sub_dark: Preprocess = lambda images, qbpm : (subtract_dark(images), qbpm)
        divide_by_qbpm: Preprocess = lambda images, qbpm : (nomalize_by_qbpm(images, qbpm), qbpm)
        def remove_by_ransac(images, qbpm):
            mask = RANSAC_regression(images.sum(axis=(1, 2)), qbpm, min_samples=2)[0]
            return images[mask], qbpm[mask]
        

        preprocessing_functions: list[Preprocess] = [
            sub_dark,
            divide_by_qbpm,
            remove_by_ransac,
            ]
        
        logger.info(f"preprocessing: subtract dark")
        logger.info(f"preprocessing: divide by qbpm")
        # logger.info(f"preprocessing: remove outlier sigma={sigma}")
        logger.info(f"preprocessing: remove outlier by ransac")
        
        rocking: RockingProcessor = RockingProcessor(preprocessing_functions, logger)
        rocking.scan(run_num)
        # rocking.save_as_mat("")
        
        mat_saver: SaverStrategy = SaverFactory.get_saver("mat")
        tif_saver: SaverStrategy = SaverFactory.get_saver("tif")
        npz_saver: SaverStrategy = SaverFactory.get_saver("npz")
        rocking.save(mat_saver)
        rocking.save(tif_saver)
        rocking.save(npz_saver)
        
        logger.info("Processing is over")