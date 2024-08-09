import os

import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
from roi_rectangle import RoiRectangle

from utils.math_util import gaussian, mul_deltaQ

import numpy.typing as npt
from typing import Mapping

class DataAnalyzer:
    def __init__(self, file: str, angle: int = 0) -> None:
        if not os.path.exists(file):
            raise FileNotFoundError(f"The file {file} was not found.")
        
        data: Mapping[str, npt.NDArray] = np.load(file)

        if "delay" not in data or "pon" not in data or "poff" not in data:
            raise ValueError("The file does not contain the required keys: 'delay', 'pon', 'poff'")
        
        self.delay: npt.NDArray = data["delay"]
        self.poff_images: npt.NDArray = data["poff"]
        self.pon_images: npt.NDArray = data["pon"]
        
        if angle:
            self.poff_images = rotate(self.poff_images, 45, axes=(1, 2), reshape=False)
            self.pon_images = rotate(self.pon_images, 45, axes=(1, 2), reshape=False)
            
    def get_summed_image(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        return: 
            pump off image, pump on image
        """
        return self.poff_images.sum(axis=0), self.pon_images.sum(axis=0)
    
    def pon_subtract_by_poff(self):
        return np.maximum(self.pon_images - self.poff_images, 0)
    
    def _roi_center_of_masses(self, roi_rect: RoiRectangle, images: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        roi_images = roi_rect.slice(images)
        height, width = roi_rect.height, roi_rect.width

        y_coords, x_coords = np.mgrid[:height, :width]

        total_mass = np.sum(roi_images, axis=(1, 2))
        x_centroids = np.sum(x_coords * roi_images, axis=(1, 2)) / total_mass
        y_centroids = np.sum(y_coords * roi_images, axis=(1, 2)) / total_mass

        return x_centroids, y_centroids
    
    def _roi_gaussian(self, roi_rect: RoiRectangle, images: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        roi_images = roi_rect.slice(images)
        height, width = roi_rect.height, roi_rect.width

        intensities = []
        com_xs = []
        com_ys = []

        for image in roi_images:
            max_y, max_x = np.unravel_index(np.argmax(image), image.shape)
            # max_y, max_x = height // 2, width // 2
            
            x: npt.NDArray = np.arange(0, width)
            y: npt.NDArray = np.arange(0, height)

            x_data = image.sum(axis=0)
            y_data = image.sum(axis=1)

            
            initial_guess_x = [x_data[max_x], max_x, (np.max(x_data) - np.min(x_data)) / 4]
            try:
                params_x = curve_fit(gaussian, x, x_data, p0=initial_guess_x)[0]
            except RuntimeError as e:
                print(e, ": x")
                params_x = [np.nan, np.nan, np.nan]    

            initial_guess_y = [y_data[max_y], max_y, (np.max(y_data) - np.min(y_data)) / 4]
            try:
                params_y = curve_fit(gaussian, y, y_data, p0=initial_guess_y)[0]
            except RuntimeError as e:
                print(e, ": y")
                params_y = [np.nan, np.nan, np.nan]

            gaussian_a_x, gaussain_com_x, gaussian_sig_x = params_x
            gaussian_a_y, gaussain_com_y, gaussian_sig_y = params_y

            gaussian_a = np.sqrt(gaussian_a_x * gaussian_a_y)
            intensities.append(gaussian_a)
            com_xs.append(gaussain_com_x)
            com_ys.append(gaussain_com_y)

        return np.stack(intensities), np.stack(com_xs), np.stack(com_ys)


    def _roi_intensities(self, roi_rect: RoiRectangle, images: npt.NDArray):
        roi_images = roi_rect.slice(images)

        return roi_images.mean(axis=(1, 2))

    def analyze_by_roi(self, roi_rect: RoiRectangle) -> pd.DataFrame:

        poff_com_x, poff_com_y = self._roi_center_of_masses(roi_rect, self.poff_images)
        poff_intensity = self._roi_intensities(roi_rect, self.poff_images)
        pon_com_x, pon_com_y = self._roi_center_of_masses(roi_rect, self.pon_images)
        pon_intensity = self._roi_intensities(roi_rect, self.pon_images)

        # poff_guassain_intensity, poff_gussian_com_x, poff_gussian_com_y = self._roi_gaussian(roi_rect, self.poff_images)
        # pon_guassain_intensity, pon_gussian_com_x, pon_gussian_com_y = self._roi_gaussian(roi_rect, self.pon_images)

        roi_df = pd.DataFrame(data={
            "poff_com_x": mul_deltaQ(poff_com_x - poff_com_x[0]),
            "poff_com_y": mul_deltaQ(poff_com_y - poff_com_y[0]),
            "poff_intensity": poff_intensity / poff_intensity[0],
            "pon_com_x": mul_deltaQ(pon_com_x - pon_com_x[0]),
            "pon_com_y": mul_deltaQ(pon_com_y - pon_com_y[0]),
            "pon_intensity": pon_intensity / pon_intensity[0],

            # "poff_gussian_com_x": poff_gussian_com_x,
            # "poff_gussian_com_y": poff_gussian_com_y,
            # "poff_guassain_intensity": poff_guassain_intensity / poff_guassain_intensity[0],

            # "pon_gussian_com_x": pon_gussian_com_x,
            # "pon_gussian_com_y": pon_gussian_com_y,
            # "pon_guassain_intensity": pon_guassain_intensity / pon_guassain_intensity[0],

        })
        
        roi_df = roi_df.set_index(self.delay)
        return roi_df


if __name__ == "__main__":
    import os
    import tifffile

    from gui.roi import select_roi_by_run_scan
    from utils.file_util import create_run_scan_directory
    from config import load_config
    from analyzer.draw_figure import (
        patch_rectangle, 
        draw_com_figure, 
        draw_intensity_figure, 
        draw_intensity_diff_figure, 
        draw_com_diff_figure
        )
    

    from typing import Optional

    ##########################################################
    # Run MeanDataProcessor
    ##########################################################
    '''
    scan: 143, 144, 145, 148, 149, 150, 151, 152, 153, 154, 155, 160, 161, 
    206, 207, 208, 209, 210, 211
    '''
    config = load_config()
    run_num: int = 143
    scan_num: int = 1
    comment: Optional[str] = None
    print(f"Run MeanDataProcessor run={run_num:0>3}")
    npz_dir: str = config.path.npz_dir

    file_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"
    if comment is not None:
        file_name += comment

    npz_file = os.path.join(npz_dir, file_name + ".npz")
    processor = DataAnalyzer(npz_file, 0)

    roi_rect = select_roi_by_run_scan(run_num, scan_num, 0)

    roi_rects = [roi_rect]
    names = ["center"]
    named_roi_rects = zip(names, roi_rects)
    data_df = processor.analyze_by_roi(named_roi_rects)[0]

    ##########################################################
    # Save Data
    ##########################################################

    image_dir = config.path.image_dir
    save_dir = create_run_scan_directory(image_dir, run_num, scan_num)

    poff_images = processor.poff_images
    pon_images = processor.pon_images

    roi_poff_images = roi_rect.slice(poff_images)
    roi_pon_images = roi_rect.slice(pon_images)

    # Save Tif
    tifffile.imwrite(os.path.join(save_dir, "poff.tif"), poff_images.astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, "pon.tif"), pon_images.astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, "roi_poff.tif"), roi_poff_images.astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, "roi_pon.tif"), roi_pon_images.astype(np.float32))

    # Save Data to CSV
    data_file = os.path.join(save_dir, "data.csv")
    data_df.to_csv(data_file)

    # Make Figure
    image_fig = patch_rectangle(np.log1p(processor.poff_images.sum(axis=0)), *roi_rect.get_coordinate())
    intensity_fig = draw_intensity_figure(data_df)
    intensity_diff_fig = draw_intensity_diff_figure(data_df)
    com_fig = draw_com_figure(data_df)
    com_diff_fig = draw_com_diff_figure(data_df)

    # Save Figure
    image_fig.savefig(os.path.join(save_dir, "log_image.png"))
    intensity_fig.savefig(os.path.join(save_dir, "delay-intensity.png"))
    intensity_diff_fig.savefig(os.path.join(save_dir, "delay-intensity_diff.png"))
    com_fig.savefig(os.path.join(save_dir, "delay-com.png"))
    com_diff_fig.savefig(os.path.join(save_dir, "delay-com_diff.png"))

    print("Mean Data Proccesing is Done.")



    
    