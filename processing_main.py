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
from src.gui.roi import get_roi_auto, get_single_images_from_hdf5
from src.utils.file_util import get_folder_list, get_run_scan_directory, get_file_list
from src.config.config import load_config, ExpConfig


logger: Logger = setup_logger()


def get_scan_nums(run_num: int, config: ExpConfig) -> list[int]:
    """Get Scan numbers from real directory"""
    run_dir: str = get_run_scan_directory(config.path.load_dir, run_num)
    scan_folders: list[str] = get_folder_list(run_dir)
    return [int(scan_dir.split("=")[1]) for scan_dir in scan_folders]


def setup_preprocessors(roi_rect: RoiRectangle) -> dict[str, ImagesQbpmProcessor]:
    """Return preprocessors"""

    remove_by_ransac_roi: ImagesQbpmProcessor = create_ransac_roi_outlier_remover(roi_rect)

    standard_preprocessor = compose(
        subtract_dark_background,
        remove_by_ransac_roi,
        normalize_images_by_qbpm,
    )

    return {
        "standard": standard_preprocessor,
    }


def process_scan(run_num: int, scan_num: int, config: ExpConfig) -> None:
    """Process Single Scan"""
    ################################
    load_dir = config.path.load_dir
    scan_dir = get_run_scan_directory(load_dir, run_num, scan_num)
    files = get_file_list(scan_dir)

    index_mode: Optional[int] = None
    if index_mode is None:
        index = len(files) // 2
    else:
        index = index_mode
    images = get_single_images_from_hdf5(run_num, scan_num, index)
    roi_rect = get_roi_auto(images.sum(axis=0))
    if roi_rect is None:
        raise ValueError(f"No ROI Rectangle Set for run={run_num}, scan={scan_num}")
    ################################

    logger.info(f"ROI rectangle: {roi_rect.to_tuple()}")
    preprocessors: dict[str, ImagesQbpmProcessor] = setup_preprocessors(roi_rect)

    for preprocessor_name in preprocessors:
        logger.info(f"preprocessor: {preprocessor_name}")

    processor: CoreProcessor = CoreProcessor(HDF5FileLoader, preprocessors, logger)
    processor.scan(scan_dir)

    file_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"
    # mat_saver: SaverStrategy = SaverFactory.get_saver("mat")
    npz_saver: SaverStrategy = SaverFactory.get_saver("npz")
    processor.save(npz_saver, file_name)

    logger.info(f"Processing run={run_num}, scan={scan_num} is complete")


def main() -> None:
    """
    60 Hz laser:
    197, 201, 202, 203, 204, 212, 213, 214, 217, 218,
    219, 220, 221, 222, 223, 228, 229, 230, 231, 234,
    235, 236, 237, 238, 241, 242, 243, 244, 246, 251,
    252, 253, 254, 255, 256, 259, 260, 261, 262, 263
    """

    config = load_config()
    run_nums: list[int] = config.runs
    logger.info(f"Runs to process: {run_nums}")

    for run_num in run_nums:
        logger.info(f"Run: {run_num}")
        scan_nums: list[int] = get_scan_nums(run_num, config)
        for scan_num in scan_nums:
            try:
                process_scan(run_num, scan_num, config)
            except Exception:
                logger.exception(f"Failed to process run={run_num}, scan={scan_num}")
                raise

    logger.info("All processing is complete")


if __name__ == "__main__":
    main()
