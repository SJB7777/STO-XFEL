from logger import AppLogger
from processor.core import CoreProcessor
from processor.loader import HDF5FileLoader
from processor.saver import SaverFactory, SaverStrategy
from preprocess.image_qbpm_pipeline import (
    subtract_dark_background,
    normalize_images_by_qbpm,
    create_ransac_roi_outlier_remover,
    ImagesQbpmProcessor
)
from gui.roi import select_roi_by_run_scan
from utils.file_util import get_folder_list, get_run_scan_directory
from config import load_config, ExperimentConfiguration

from roi_rectangle import RoiRectangle
from typing import Optional


logger: AppLogger = AppLogger("MainProcessor")

def get_scan_nums(run_num: int, config: ExperimentConfiguration) -> list[int]:
    run_dir: str = get_run_scan_directory(config.path.load_dir, run_num)
    scan_folders: list[str] = get_folder_list(run_dir)
    return [int(scan_dir.split("=")[1]) for scan_dir in scan_folders]

def setup_pipelines(roi_rect: RoiRectangle) -> dict[str, list[ImagesQbpmProcessor]]:
    remove_by_ransac_roi: ImagesQbpmProcessor = create_ransac_roi_outlier_remover(roi_rect)

    standard_pipeline = [
        subtract_dark_background,
        remove_by_ransac_roi,
        normalize_images_by_qbpm,
    ]

    return {
        "standard" : standard_pipeline,
    }
    
def process_scan(run_num: int, scan_num: int, config: ExperimentConfiguration) -> None:
    load_dir = config.path.load_dir
    scan_dir = get_run_scan_directory(load_dir, run_num, scan_num)

    roi_rect: Optional[RoiRectangle] = select_roi_by_run_scan(run_num, scan_num)
    if roi_rect is None:
        raise ValueError(f"No ROI Rectangle Set for run={run_num}, scan={scan_num}")

    logger.info(f"ROI rectangle: {roi_rect.get_coordinate()}")
    pipelines: dict[str, list[ImagesQbpmProcessor]] = setup_pipelines(roi_rect)

    for pipeline_name, pipeline in pipelines.items():
        logger.info(f"PipeLine: {pipeline_name}")
        for function in pipeline:
            logger.info(f"preprocess: {function.__name__}")

    processor: CoreProcessor = CoreProcessor(HDF5FileLoader, {"standard": pipeline}, logger)
    processor.scan(scan_dir)

    file_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"
    mat_saver: SaverStrategy = SaverFactory.get_saver("mat")
    processor.save(mat_saver, file_name)

    logger.info(f"Processing run={run_num}, scan={scan_num} is complete")

def main() -> None:
    """
    60 Hz laser:
    197, 201, 202, 203, 204, 212, 213, 214, 217, 218, 
    219, 220, 221, 222, 223, 228, 229, 230, 231, 234, 235, 236, 237, 238, 241, 242, 243, 244, 246, 251,
    252, 253, 254, 255, 256, 259, 260, 261, 262, 263
    """

    config = load_config()
    run_nums: list[int] = [197]
    logger.info(f"Runs to process: {run_nums}")

    for run_num in run_nums:
        scan_nums: list[int] = get_scan_nums(run_num, config)
        for scan_num in scan_nums:
            try:
                process_scan(run_num, scan_num, config)
            except Exception as e:
                logger.error(f"Failed to process run={run_num}, scan={scan_num}: {type(e).__name__}: {str(e)}")
                import traceback
                error_message = traceback.format_exc()
                logger.error("\n" + error_message)

    logger.info("All processing is complete")

if __name__ == "__main__":
    main()