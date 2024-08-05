from logger import AppLogger
from core.raw_data_processor import RawDataProcessor
from core.loader_strategy import HDF5FileLoader
from core.saver import SaverFactory, SaverStrategy
from preprocess.image_qbpm_pipeline import (
    ImagesQbpmProcessor,
    subtract_dark_background,
    normalize_images_by_qbpm,
    create_ransac_roi_outlier_remover,
    shift_to_positive
)
from gui.roi import select_roi_by_run_scan
from utils.file_util import get_folder_list, get_run_scan_directory
from config import load_config

from roi_rectangle import RoiRectangle
from typing import Optional

logger: AppLogger = AppLogger("MainProcessor")

def get_scan_nums(run_num: int) -> list[tuple[int, int]]:
    config = load_config()
    run_dir: str = get_run_scan_directory(config.path.load_dir, run_num)
    scan_folders: list[str] = get_folder_list(run_dir)

    return [int(scan_dir.split("=")[1]) for scan_dir in scan_folders]

def processing(run_num: int, scan_num: int) -> None:
    config = load_config()
    load_dir = config.path.load_dir
    scan_dir = get_run_scan_directory(load_dir, run_num, scan_num)

    roi_rect: Optional[RoiRectangle] = select_roi_by_run_scan(run_num, scan_num)
    # roi_rect = RoiRectangle(126, 60, 161, 99)
    
    if roi_rect is None:
        raise Exception(f"No Roi Rectangle Setted: roi_rect is None")
    
    logger.info(f"roi rectangle: {roi_rect.get_coordinate()}")
    remove_by_ransac_roi: ImagesQbpmProcessor = create_ransac_roi_outlier_remover(roi_rect)

    # Standard Pipeline
    standard_pipeline: list[ImagesQbpmProcessor] = [
        subtract_dark_background,
        remove_by_ransac_roi,
        normalize_images_by_qbpm,
    ]
    
    # No Dark
    no_dark_pipeline: list[ImagesQbpmProcessor] = [
        remove_by_ransac_roi,
        normalize_images_by_qbpm,
    ]
    
    # No Dark
    add_bias_pipeline: list[ImagesQbpmProcessor] = [
        shift_to_positive
    ]
    
    # Empty Pipeline
    empty_pipeline: list[ImagesQbpmProcessor] = []
    
    # Dict of Pipelines
    pipelines: dict[str, list[ImagesQbpmProcessor]] = {
        "empty" : empty_pipeline,
        "add bias" : add_bias_pipeline,
    }

    for pipeline_name, pipeline in pipelines.items():
        logger.info(f"PipeLine: {pipeline_name}")
        for function in pipeline:
            logger.info(f"preprocess: {function.__name__}")

    rdp = RawDataProcessor(HDF5FileLoader, pipelines, logger)
    rdp.scan(scan_dir)
    mat_saver: SaverStrategy = SaverFactory.get_saver("mat")

    file_name = f"run={run_num:0>4}_scan={scan_num:0>4}"
    rdp.save(mat_saver, file_name)
    
    logger.info(f"Processing run={run_num} is over")

def main() -> None:

    run_nums: list[int] = [1]
    logger.info(f"run: {run_nums}")

    for run_num in run_nums:
        scan_nums: list[int] = get_scan_nums(run_num)
        for scan_num in scan_nums:
            try:
                processing(run_num, scan_num)
            except Exception as e:
                logger.error(f"Failed to Process: {type(e)}: {str(e)}")
                import traceback
                error_message = traceback.format_exc()
                logger.error("\n" + error_message)
                return None

    logger.info("Processing is over")


if __name__ == "__main__":
    main()