import os
from collections import defaultdict
from typing import Optional, DefaultDict, Type, Any
import json

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.utils.file_util import get_file_list
from src.processor.saver import SaverStrategy
from src.processor.loader import RawDataLoader
from src.preprocessor.image_qbpm_preprocessor import ImagesQbpmProcessor
from src.logger import setup_logger, Logger
from src.config import load_config, ExperimentConfiguration


class CoreProcessor:
    """
    Use ETL Pattern
    """
    def __init__(
        self,
        LoaderStrategy: Type[RawDataLoader],  # pylint: disable=invalid-name
        preprocessor: Optional[dict[str, ImagesQbpmProcessor]] = None,
        logger: Optional[Logger] = None
    ) -> None:

        self.LoaderStrategy: Type[RawDataLoader] = LoaderStrategy  # pylint: disable=invalid-name
        self.preprocessor: dict[str, ImagesQbpmProcessor] = preprocessor if preprocessor is not None else {"no_processing": lambda x: x}
        self.preprocessor_data_dict: dict[str, DefaultDict[str, list]] = {pipline_name: defaultdict(list) for pipline_name in self.preprocessor}

        self.logger: Logger = logger if logger is not None else setup_logger("MainProcessor")
        self.config: ExperimentConfiguration = load_config()
        self.result: dict[str, DefaultDict[str, npt.NDArray]] = {}
        config_dict_jump: str = json.dumps(self.config.to_config_dict(), indent=4)
        self.logger.info(f"Meta Data:\n{config_dict_jump}")

    def scan(self, scan_dir: str):
        """
        Scans directories for rocking scan data and processes them.

        Parameters:
        - run_num (int): Run number to scan.
        """

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

        for hdf5_file in pbar:

            loader_strategy = self.get_loader(scan_dir, hdf5_file)
            if loader_strategy is not None:
                self.add_processed_data_to_dict(loader_strategy)

        return self.stack_processed_data(self.preprocessor_data_dict)

    def get_loader(self, scan_dir: str, hdf5_file: str) -> Optional[RawDataLoader]:
        """Get Loader"""
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
        #     self.logger.exception(f"Failed to load: {type(e)}: {str(e)}")
        #     import traceback
        #     error_message = traceback.format_exc()
        #     self.logger.exception(error_message)
        #     return None

    def add_processed_data_to_dict(self, loader_strategy: RawDataLoader) -> dict[str, DefaultDict[str, list]]:

        preprocessor_data: dict[str, dict[str, Any]] = {}
        for preprocessor_name, preprocessor in self.preprocessor.items():

            data: dict[str, Any] = {}

            images_dict = loader_strategy.get_data()
            if "pon" in images_dict:
                applied_images: npt.NDArray = preprocessor((images_dict['pon'], images_dict['pon_qbpm']))[0]
                data['pon'] = applied_images.mean(axis=0)
            if 'poff' in images_dict:
                applied_images: npt.NDArray = preprocessor((images_dict['poff'], images_dict['poff_qbpm']))[0]
                data['poff'] = applied_images.mean(axis=0)

            data["delay"] = loader_strategy.delay
            preprocessor_data[preprocessor_name] = data

        for preprocessor_name, data in preprocessor_data.items():
            for data_name, data_value in data.items():
                self.preprocessor_data_dict[preprocessor_name][data_name].append(data_value)

    def stack_processed_data(self, preprocessor_data_dict: dict[str, DefaultDict[str, list]]) -> dict[str, DefaultDict[str, npt.NDArray]]:
        preprocessor_data_dict_result: dict[str, DefaultDict[str, npt.NDArray]] = {}
        for preprocessor_name, data in preprocessor_data_dict.items():
            preprocessor_data_dict_result[preprocessor_name] = {data_name: np.stack(data_list) for data_name, data_list in data.items()}

        return preprocessor_data_dict_result

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
            raise ValueError("Nothing to save")

        for pipline_name, data_dict in self.result.items():
            file_base_name = f"{file_name}"

            saver.save(file_base_name, data_dict)
            self.logger.info(f"Finished preprocessor: {pipline_name}")
            self.logger.info(f"Data Dict Keys: {data_dict.keys()}")
            self.logger.info(f"Saved file '{saver.file}'")


if __name__ == "__main__":

    from src.processor.loader import HDF5FileLoader
    from src.processor.saver import SaverFactory
    from src.preprocessor.image_qbpm_preprocessor import (
        compose,
        subtract_dark_background,
        normalize_images_by_qbpm,
        remove_outliers_using_ransac,
        equalize_intensities
    )

    run_num: int = 1
    scan_num: int = 1
    logger: Logger = setup_logger()

    # preprocessor 1
    preprocessor_normalize_images_by_qbpm: ImagesQbpmProcessor = compose(
        subtract_dark_background,
        remove_outliers_using_ransac,
        normalize_images_by_qbpm,
    )
    logger.info("preprocessor: normalize_images_by_qbpm")
    logger.info("preprocessing: subtract_dark_background")
    logger.info("preprocessing: remove_by_ransac")
    logger.info("preprocessing: normalize_images_by_qbpm")

    # preprocessor 2
    preprocessor_equalize_intensities: ImagesQbpmProcessor = compose(
        subtract_dark_background,
        remove_outliers_using_ransac,
        equalize_intensities,
    )
    logger.info("preprocessor: normalize_images_by_qbpm")
    logger.info("preprocessing: subtract_dark_background")
    logger.info("preprocessing: remove_by_ransac")
    logger.info("preprocessing: equalize_intensities")

    # preprocessor 3
    preprocessor_no_normalize: ImagesQbpmProcessor = compose(
        subtract_dark_background,
        remove_outliers_using_ransac,
    )
    logger.info("preprocessor: no_normalize")
    logger.info("preprocessing: subtract_dark_background")
    logger.info("preprocessing: remove_by_ransac")

    preprocessors: dict[str, ImagesQbpmProcessor] = {
        "normalize_images_by_qbpm": preprocessor_normalize_images_by_qbpm,
        "equalize_intensities": preprocessor_equalize_intensities,
        "no_normalize": preprocessor_no_normalize
    }

    cp = CoreProcessor(HDF5FileLoader, preprocessors, logger)
    cp.scan(run_num)

    file_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"

    mat_saver: SaverStrategy = SaverFactory.get_saver("mat")
    # tif_saver: SaverStrategy = SaverFactory.get_saver("tif")
    # npz_saver: SaverStrategy = SaverFactory.get_saver("npz")
    cp.save(mat_saver, file_name)
    # cp.save(tif_saver)
    # cp.save(npz_saver)

    logger.info("Processing is over")
