from roi_rectangle import RoiRectangle

from logger import AppLogger
from core.raw_data_processor import RawDataProcessor
from core.loader_strategy import HDF5FileLoader
from core.saver import SaverFactory, SaverStrategy
from preprocess.image_qbpm_processors import (
    ImageQbpmProcessor,
    subtract_dark_background,
    normalize_images_by_qbpm,
    remove_by_ransac,
    equalize_intensities,
    create_remove_by_ransac_roi
)
from gui.roi import select_roi_by_run_scan

from roi_rectangle import RoiRectangle

logger: AppLogger = AppLogger("MainProcessor")

run_nums = [1]
logger.info(f"run: {run_nums}")
for run_num in run_nums:
    roi_rect: RoiRectangle = select_roi_by_run_scan(run_num, 1)
    remove_by_ransac_roi: ImageQbpmProcessor = create_remove_by_ransac_roi(roi_rect)

    preprocessing_functions: list[ImageQbpmProcessor] = [
        subtract_dark_background,
        remove_by_ransac_roi,
        normalize_images_by_qbpm,
    ]

    logger.info(f"preprocessing: subtract_dark_background")
    logger.info(f"preprocessing: remove_by_ransac_roi {roi_rect.get_coordinate()}")
    logger.info(f"preprocessing: normalize_images_by_qbpm")
    
    cp = RawDataProcessor(HDF5FileLoader, preprocessing_functions, logger)
    cp.scan(run_num)

    mat_saver: SaverStrategy = SaverFactory.get_saver("mat")
    tif_saver: SaverStrategy = SaverFactory.get_saver("tif")
    npz_saver: SaverStrategy = SaverFactory.get_saver("npz")

    cp.save(mat_saver)
    cp.save(tif_saver)
    cp.save(npz_saver)
    
    logger.info(f"Processing run={run_num} is over")

logger.info("Processing is over")