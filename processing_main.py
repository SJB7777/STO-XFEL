import os
from typing import Optional

from roi_rectangle import RoiRectangle

from src.logger import setup_logger, Logger
from src.processor.core import CoreProcessor
from src.processor.loader import HDF5FileLoader
from src.processor.saver import SaverFactory, SaverStrategy
from src.preprocessor.image_qbpm_preprocessor import (
    compose,
    subtract_dark_background,
    normalize_images_by_qbpm,
    create_ransac_roi_outlier_remover,
    ImagesQbpmProcessor
)
from src.gui.roi import get_roi_auto, get_hdf5_images
from src.utils.file_util import get_folder_list, get_run_scan_directory, get_file_list
from src.config.config import load_config, ExpConfig


def get_scan_nums(run_num: int, config: ExpConfig) -> list[int]:
    """Get Scan numbers from real directory"""
    run_dir: str = get_run_scan_directory(config.path.load_dir, run_num)
    scan_folders: list[str] = get_folder_list(run_dir)
    return [int(scan_dir.split("=")[1]) for scan_dir in scan_folders]


def get_roi(scan_dir: str, config: ExpConfig, index_mode: Optional[int] = None) -> RoiRectangle:
    """Get Roi for QBPM Normalization"""
    files = get_file_list(scan_dir)

    if index_mode is None:
        index = len(files) // 2
    else:
        index = index_mode

    file: str = os.path.join(scan_dir, files[index])
    image = get_hdf5_images(file, config).sum(axis=0)
    return get_roi_auto(image)


def setup_preprocessors(roi_rect: RoiRectangle) -> dict[str, ImagesQbpmProcessor]:
    """Return preprocessors"""

    remove_by_ransac_roi: ImagesQbpmProcessor = create_ransac_roi_outlier_remover(roi_rect)

    standard_preprocessor = compose(
        # subtract_dark_background,
        remove_by_ransac_roi,
        normalize_images_by_qbpm,
    )

    return {
        "standard": standard_preprocessor,
    }


def process_scan(run_num: int, scan_num: int, config: ExpConfig, logger: Logger) -> None:
    """Process Single Scan"""

    load_dir = config.path.load_dir
    scan_dir = get_run_scan_directory(load_dir, run_num, scan_num)

    roi_rect = get_roi(scan_dir, config, 0)
    if roi_rect is None:
        raise ValueError(f"No ROI Rectangle Set for run={run_num}, scan={scan_num}")
    logger.info(f"ROI rectangle: {roi_rect.to_tuple()}")
    preprocessors: dict[str, ImagesQbpmProcessor] = setup_preprocessors(roi_rect)

    for preprocessor_name in preprocessors:
        logger.info(f"preprocessor: {preprocessor_name}")
    processor: CoreProcessor = CoreProcessor(HDF5FileLoader, scan_dir, preprocessors, logger)

    file_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"

    # Set SaverStrategy
    mat_saver: SaverStrategy = SaverFactory.get_saver("mat")
    processor.save(mat_saver, file_name)

    npz_saver: SaverStrategy = SaverFactory.get_saver("npz")
    processor.save(npz_saver, file_name)

    logger.info(f"Processing run={run_num}, scan={scan_num} is complete")


def main() -> None:
    """Processing"""
    logger: Logger = setup_logger()

    config = load_config()
    run_nums: list[int] = config.runs
    logger.info(f"Runs to process: {run_nums}")

    for run_num in run_nums: # pylint: disable=not-an-iterable
        logger.info(f"Run: {run_num}")
        scan_nums: list[int] = get_scan_nums(run_num, config)
        for scan_num in scan_nums:
            try:
                process_scan(run_num, scan_num, config, logger)
            except Exception:
                logger.exception(f"Failed to process run={run_num}, scan={scan_num}")
                raise

    logger.info("All processing is complete")


if __name__ == "__main__":
    main()
