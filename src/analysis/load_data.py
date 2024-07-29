from roi_rectangle import RoiRectangle
import numpy as np
from scipy.ndimage import center_of_mass

import numpy.typing as npt


class ReadDelay:
    def __init__(self, file: str) -> None:
        try:
            data = np.load(file)
            if "delay" not in data or "pon" not in data and "poff" not in data:
                raise ValueError("The file does not contain the required keys: 'delay', 'pon', 'poff'")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file} was not found.")
        
        self.delay = data["delay"]
        self.poff_images = data["poff"]
        self.pon_images = data["pon"]
        
    def _analyze_roi(self, roi_rect: RoiRectangle, images: npt.NDArray):
        roi_images = roi_rect.slice(images)
        num_images, height, width = roi_images.shape

        
        y_coords, x_coords = np.mgrid[:height, :width]

        total_mass = np.sum(roi_images, axis=(1, 2))
        x_centroids = np.sum(x_coords * roi_images, axis=(1, 2)) / total_mass
        y_centroids = np.sum(y_coords * roi_images, axis=(1, 2)) / total_mass

        centroids = np.stack((x_centroids, y_centroids), axis=-1)

    def tracking_by_rois(self, roi_rects: list[RoiRectangle]):
        for roi_rect in roi_rects:
            pass
