import os
from collections import defaultdict
from typing import Optional, DefaultDict, Type, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from cuptlib_config.palxfel import load_palxfel_config

from utils.file_util import get_run_scan_directory, get_folder_list, get_file_list
from scan.saver import SaverStrategy
from scan.loading_strategy import HDF5LoadingStrategy
from logger import AppLogger
from preprocess.image_qbpm_processors import ImageQbpmProcessor


class CoreProcesser:
    
    def __init__(self, scan_strategy_class: Type[HDF5LoadingStrategy], preprocessing_functions: Optional[list[ImageQbpmProcessor]] = None, logger: Optional[AppLogger] = None) -> None:
        self.ScanStrategy = scan_strategy_class
        self.preprocessing_functions = preprocessing_functions if preprocessing_functions is not None else []
        self.logger = logger if logger is not None else AppLogger("RockingProcessor")
        
        self.config = load_palxfel_config("config.ini")
        self.logger.add_metadata(self.config.to_config_dict())
        self.data_dict: dict[tuple[int, int], dict[str, npt.NDArray]] = {}

    def scan(self, run_num: int):
        """
        Scans directories for rocking scan data and processes them.

        Parameters:
        - run_num (int): Run number to scan.
        """
        self.logger.info(f"Starting scan for run number: {run_num}")
        self.data_dict.clear()

        root_dir = self.config.path.load_dir
        run_dir = get_run_scan_directory(root_dir, run_num)

        scan_folders = get_folder_list(run_dir)
        for scan_folder in scan_folders:
            scan_num = int(scan_folder.split("=")[1])
            scan_dir = os.path.join(run_dir, scan_folder)
            self.logger.info(f"Processing scan folder: {scan_folder}")
            
            data_dict = self._single_scan(scan_dir)
            run_scan: tuple[int, int] = (run_num, scan_num)
            self.data_dict[run_scan] = data_dict
            
            self.logger.info(f"Completed processing for run={run_num}, scan={scan_num}")
            
    def _single_scan(self, scan_dir: str) -> dict[str, npt.NDArray]:
        """
        Processes a single scan directory.

        Parameters:
        - scan_dir (str): Directory path of the scan to process.

        Returns:
        - dict[str, npt.NDArray]: Dictionary containing stacked images from the scan.
        """
        self.logger.info(f"Starting single scan for directory: {scan_dir}")
        data_list_dict: DefaultDict[str, list] = defaultdict(list)
        
        hdf5_files = get_file_list(scan_dir)
        pbar = tqdm(hdf5_files, total=len(hdf5_files))
        for hdf5_file in pbar:
            hdf5_dir = os.path.join(scan_dir, hdf5_file)
            
            try:
                rr: HDF5LoadingStrategy = self.ScanStrategy(hdf5_dir)
            except KeyError as e:
                self.logger.warning(f"{e}")
                self.logger.warning(f"KeyError happened in {scan_dir}")
                continue
            # except Exception as e:
            #     self.logger.error(f"Failed to load frame {i}: {type(e)}: {str(e)}")

            #     import traceback
            #     traceback.print_exc()
            #     continue

            rr.apply_preprocessing_functions(self.preprocessing_functions)
            
            data_list_dict_temp = rr.get_data()
            for key, val in data_list_dict_temp.items():
                data_list_dict[key].append(val)
        
        self.logger.info(f"Completed single scan for directory: {scan_dir}")
        
        data_dict: dict[str, npt.NDArray] = {key: np.stack(value) for key, value in data_list_dict.items()}
        return data_dict
    
    def save(self, saver: SaverStrategy, comment: str = ""):
        """
        Saves processed images using a specified saving strategy.

        Parameters:
        - saver (SaverStrategy): Saving strategy to use.
        - comment (str, optional): Comment to append to the file name.
        """
        self.logger.info(f"Start to save as {saver.file_type.capitalize()}")
        
        if not self.data_dict:
            self.logger.error("Nothing to save")
            raise Exception("Nothing to save")
        
        for (run, scan), data_dict in self.data_dict.items():
            file_base_name = f"run={run:0>4}_scan={scan:0>4}"
            saver.save(file_base_name, data_dict, comment)
            self.logger.info(f"Data Dict Keys: {data_dict.keys()}")        
            self.logger.info(f"Saved file '{saver.file}'")

if __name__ == "__main__":
    
    from scan.loading_strategy import HDF5FileLoader
    from scan.saver import SaverFactory
    from preprocess.image_qbpm_processors import (
        subtract_dark_background,
        normalize_images_by_qbpm,
        remove_by_ransac,
        equalize_intensities
    )
    run_num = 1
    logger: AppLogger = AppLogger("RockingProcessor")
    
    preprocessing_functions: list[ImageQbpmProcessor] = [
        subtract_dark_background,
        normalize_images_by_qbpm,
        remove_by_ransac,
    ]
    
    logger.info(f"preprocessing: subtract dark")
    logger.info(f"preprocessing: divide by qbpm")
    logger.info(f"preprocessing: remove outlier by ransac")
    
    cp = CoreProcesser(HDF5FileLoader, preprocessing_functions, logger)
    cp.scan(run_num)
    
    mat_saver: SaverStrategy = SaverFactory.get_saver("mat")
    tif_saver: SaverStrategy = SaverFactory.get_saver("tif")
    npz_saver: SaverStrategy = SaverFactory.get_saver("npz")
    cp.save(mat_saver)
    cp.save(tif_saver)
    cp.save(npz_saver)
    
    logger.info("Processing is over")