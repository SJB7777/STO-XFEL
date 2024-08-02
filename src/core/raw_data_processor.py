import os
from collections import defaultdict
from typing import Optional, DefaultDict, Type, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from cuptlib_config.palxfel import load_palxfel_config

from utils.file_util import get_run_scan_directory, get_folder_list, get_file_list
from core.saver import SaverStrategy
from core.loader_strategy import HDF5LoaderInterface
from logger import AppLogger

from typing import Any
from preprocess.image_qbpm_processors import ImagesQbpmProcessor


class RawDataProcessor:
    
    def __init__(self, LoaderStrategy: Type[HDF5LoaderInterface], pipelines: Optional[dict[str, list[ImagesQbpmProcessor]]] = None, logger: Optional[AppLogger] = None) -> None:
        
        self.LoaderStrategy = LoaderStrategy
        self.pipelines = pipelines if pipelines is not None else {"no_processing" : []}

        self.logger = logger if logger is not None else AppLogger("MainProcessor")
        self.config = load_palxfel_config("config.ini")
        self.logger.add_metadata(self.config.to_config_dict())

    def scan(self, scan_dir: str):
        """
        Scans directories for rocking scan data and processes them.

        Parameters:
        - run_num (int): Run number to scan.
        """
        scan_dir = scan_dir
        self.logger.info(f"Starting scan: {scan_dir}")

        self.result: dict[str, DefaultDict[str, npt.NDArray]] = self.process_scan_directory(scan_dir)
        self.logger.info(f"Completed processing: {scan_dir}")
            
    def process_scan_directory(self, scan_dir: str) -> dict[str, DefaultDict[str, npt.NDArray]]:
        """
        Processes a single scan directory.

        Parameters:
        - scan_dir (str): Directory path of the scan to process.

        Returns:
        - dict[str, npt.NDArray]: Dictionary containing stacked images from the scan.
        """
        self.logger.info(f"Starting single scan for directory: {scan_dir}")

        hdf5_files = get_file_list(scan_dir)
        pbar = tqdm(hdf5_files, total=len(hdf5_files))

        self.pipeline_data_dict: dict[str, DefaultDict[str, list]] = {pipline_name : defaultdict(list) for pipline_name in self.pipelines}

        for hdf5_file in pbar:

            loader_strategy = self.get_loader_strategy(scan_dir, hdf5_file)
            if loader_strategy is not None:
                self.add_processed_data_to_dict(loader_strategy)

        return self.stack_processed_data(self.pipeline_data_dict)
    
    def get_loader_strategy(self, scan_dir: str, hdf5_file: str) -> Optional[HDF5LoaderInterface]:
        hdf5_dir = os.path.join(scan_dir, hdf5_file)
        try:

            return self.LoaderStrategy(hdf5_dir)
        
        except KeyError as e:
            self.logger.warning(f"{e}")
            self.logger.warning(f"KeyError happened in {scan_dir}")
            return None
        
        except FileNotFoundError as e:
            self.logger.warning(f"{e}")
            self.logger.warning(f"FileNotFoundError happened in {scan_dir}")
            return None
        
        # except Exception as e:
        #     self.logger.error(f"Failed to load: {type(e)}: {str(e)}")

        #     import traceback
        #     traceback.print_exc()

        #     return None

    def apply_pipeline(self, pipeline: list[ImagesQbpmProcessor], images: npt.NDArray, qbpm: npt.NDArray) -> npt.NDArray:
        
        for function in pipeline:
            images, qbpm = function(images, qbpm)
        
        return images
    
    def add_processed_data_to_dict(self, loader_strategy: HDF5LoaderInterface) -> dict[str, DefaultDict[str, list]]:

        pipeline_data:dict[str, dict[str, Any]] = {}
        for pipeline_name, pipeline in self.pipelines.items():
            data: dict[str, Any] = {}

            # for image_name, images in loader_strategy.get_images_dict().items():
            #     applied_images: npt.NDArray = self.apply_pipeline(pipeline, images, loader_strategy.qbpm_sum)
            #     data[image_name] = applied_images.mean(axis=0)
            images_dict = loader_strategy.get_images_dict()
            if "pon" in images_dict:
                applied_images: npt.NDArray = self.apply_pipeline(pipeline, images_dict['pon'], images_dict['pon_qbpm'])
                data['pon'] = applied_images.mean(axis=0)
            if 'poff' in images_dict:
                applied_images: npt.NDArray = self.apply_pipeline(pipeline, images_dict['poff'], images_dict['poff_qbpm'])
                data['poff'] = applied_images.mean(axis=0)
                
            data["delay"] = loader_strategy.delay
            pipeline_data[pipeline_name] = data
        
        for pipeline_name, data in pipeline_data.items():
            for data_name, data_value in data.items():
                self.pipeline_data_dict[pipeline_name][data_name].append(data_value)

    def stack_processed_data(self, pipeline_data_dict: dict[str, DefaultDict[str, list]]) -> dict[str, DefaultDict[str, npt.NDArray]]:
        pipeline_data_dict_result: dict[str, DefaultDict[str, npt.NDArray]] = {}
        for pipeline_name, data in pipeline_data_dict.items():
            pipeline_data_dict_result[pipeline_name] = {data_name: np.stack(data_list) for data_name, data_list in data.items()}

        return pipeline_data_dict_result
    

    def save(self, saver: SaverStrategy, file_name: str):
        """
        Saves processed images using a specified saving strategy.

        Parameters:
        - saver (SaverStrategy): Saving strategy to use.
        - comment (str, optional): Comment to append to the file name.
        """
        self.logger.info(f"Start to save as {saver.file_type.capitalize()}")
        
        if not self.result:
            self.logger.error("Nothing to save")
            raise Exception("Nothing to save")
        
        for pipline_name, data_dict in self.result.items():
            file_base_name = f"{file_name}"
            saver.save(file_base_name, data_dict)
            self.logger.info(f"Finished Pipeline: {pipline_name}")
            self.logger.info(f"Data Dict Keys: {data_dict.keys()}")        
            self.logger.info(f"Saved file '{saver.file}'")

if __name__ == "__main__":
    
    from core.loader_strategy import HDF5FileLoader
    from core.saver import SaverFactory
    from preprocess.image_qbpm_processors import (
        subtract_dark_background,
        normalize_images_by_qbpm,
        remove_by_ransac,
        equalize_intensities
    )
    run_num = 1
    logger: AppLogger = AppLogger("MainProcessor")
    
    # Pipeline 1
    pipeline_normalize_images_by_qbpm: list[ImagesQbpmProcessor] = [
        subtract_dark_background,
        remove_by_ransac,
        normalize_images_by_qbpm,
    ]
    logger.info(f"Pipeline: normalize_images_by_qbpm")
    logger.info(f"preprocessing: subtract_dark_background")
    logger.info(f"preprocessing: remove_by_ransac")
    logger.info(f"preprocessing: normalize_images_by_qbpm")
    
    # Pipeline 2
    pipeline_equalize_intensities: list[ImagesQbpmProcessor] = [
        subtract_dark_background,
        remove_by_ransac,
        equalize_intensities,
    ]
    logger.info(f"Pipeline: normalize_images_by_qbpm")
    logger.info(f"preprocessing: subtract_dark_background")
    logger.info(f"preprocessing: remove_by_ransac")
    logger.info(f"preprocessing: equalize_intensities")

    # Pipeline 3
    pipeline_no_normalize: list[ImagesQbpmProcessor] = [
        subtract_dark_background,
        remove_by_ransac,
    ]
    logger.info(f"Pipeline: no_normalize")
    logger.info(f"preprocessing: subtract_dark_background")
    logger.info(f"preprocessing: remove_by_ransac")

    pipelines: dict[str, list[ImagesQbpmProcessor]] = {
        "normalize_images_by_qbpm" : pipeline_normalize_images_by_qbpm,
        "equalize_intensities": pipeline_equalize_intensities,
        "no_normalize" : pipeline_no_normalize
    }

    cp = RawDataProcessor(HDF5FileLoader, pipelines, logger)
    cp.scan(run_num)
    
    mat_saver: SaverStrategy = SaverFactory.get_saver("mat")
    # tif_saver: SaverStrategy = SaverFactory.get_saver("tif")
    # npz_saver: SaverStrategy = SaverFactory.get_saver("npz")
    cp.save(mat_saver)
    # cp.save(tif_saver)
    # cp.save(npz_saver)
    
    logger.info("Processing is over")