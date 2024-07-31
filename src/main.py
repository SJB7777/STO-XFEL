import os

from logger import AppLogger
from core.raw_data_processor import RawDataProcessor
from core.loader_strategy import HDF5FileLoader
from core.saver import SaverFactory, SaverStrategy
from preprocess.image_qbpm_processors import (
    ImagesQbpmProcessor,
    subtract_dark_background,
    normalize_images_by_qbpm,
    remove_by_ransac,
    equalize_intensities,
    create_remove_by_ransac_roi
)
from gui.roi import select_roi_by_run_scan
from utils.file_util import get_folder_list, get_run_scan_directory

from roi_rectangle import RoiRectangle
from cuptlib_config.palxfel import load_palxfel_config

logger: AppLogger = AppLogger("MainProcessor")

def get_scan_nums(run_num: int) -> list[tuple[int, int]]:
    config = load_palxfel_config("config.ini")
    run_dir: str = get_run_scan_directory(config.path.load_dir, run_num)
    scan_folders: list[str] = get_folder_list(run_dir)

    return [int(scan_dir.split("=")[1]) for scan_dir in scan_folders]

def processing(run_num: int, scan_num: int) -> None:
    config = load_palxfel_config("config.ini")
    load_dir = config.path.load_dir
    scan_dir = get_run_scan_directory(load_dir, run_num, scan_num)

    roi_rect: RoiRectangle = select_roi_by_run_scan(run_num, scan_num)
    logger.info(f"roi rectangle: {roi_rect.get_coordinate()}")
    remove_by_ransac_roi: ImagesQbpmProcessor = create_remove_by_ransac_roi(roi_rect)

    # Pipeline 1
    pipeline_normalize_images_by_qbpm: list[ImagesQbpmProcessor] = [
        subtract_dark_background,
        remove_by_ransac_roi,
        normalize_images_by_qbpm,
    ]
    logger.info(f"Pipeline: normalize_images_by_qbpm")
    logger.info(f"preprocessing: subtract_dark_background")
    logger.info(f"preprocessing: remove_by_ransac_roi")
    logger.info(f"preprocessing: normalize_images_by_qbpm")
    
    # Pipeline 2
    pipeline_equalize_intensities: list[ImagesQbpmProcessor] = [
        subtract_dark_background,
        remove_by_ransac_roi,
        equalize_intensities,
    ]
    logger.info(f"Pipeline: normalize_images_by_qbpm")
    logger.info(f"preprocessing: subtract_dark_background")
    logger.info(f"preprocessing: remove_by_ransac_roi")
    logger.info(f"preprocessing: equalize_intensities")

    # Pipeline 3
    pipeline_no_normalize: list[ImagesQbpmProcessor] = [
        subtract_dark_background,
        remove_by_ransac_roi,
    ]
    logger.info(f"Pipeline: no_normalize")
    logger.info(f"preprocessing: subtract_dark_background")
    logger.info(f"preprocessing: remove_by_ransac_roi")

    pipelines: dict[str, list[ImagesQbpmProcessor]] = {
        "normalize_images_by_qbpm" : pipeline_normalize_images_by_qbpm,
        "equalize_intensities": pipeline_equalize_intensities,
        "no_normalize" : pipeline_no_normalize
    }

    cp = RawDataProcessor(HDF5FileLoader, pipelines, logger)
    cp.scan(scan_dir)
    mat_saver: SaverStrategy = SaverFactory.get_saver("mat")

    file_name = f"{run_num:0>4}_{scan_num:0>4}"
    cp.save(mat_saver, file_name)
    
    logger.info(f"Processing run={run_num} is over")


run_nums: list[int] = [176]
logger.info(f"run: {run_nums}")

for run_num in run_nums:
    scan_nums: list[int] = get_scan_nums(run_num)
    for scan_num in scan_nums:
        processing(run_num, scan_num)

logger.info("Processing is over")