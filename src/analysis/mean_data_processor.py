from roi_rectangle import RoiRectangle
import numpy as np
import pandas as pd

import numpy.typing as npt


class MeanDataProcessor:
    def __init__(self, file: str) -> None:
        try:
            data = np.load(file)
            if "delay" not in data or "pon" not in data or "poff" not in data:
                raise ValueError("The file does not contain the required keys: 'delay', 'pon', 'poff'")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file} was not found.")

        self.delay = data["delay"]
        self.poff_images = data["poff"]
        self.pon_images = data["pon"]
        
    def _roi_center_of_masses(self, roi_rect: RoiRectangle, images: npt.NDArray):
        roi_images = roi_rect.slice(images)
        num_images, height, width = roi_images.shape

        y_coords, x_coords = np.mgrid[:height, :width]

        total_mass = np.sum(roi_images, axis=(1, 2))
        x_centroids = np.sum(x_coords * roi_images, axis=(1, 2)) / total_mass
        y_centroids = np.sum(y_coords * roi_images, axis=(1, 2)) / total_mass

        return np.stack((x_centroids, y_centroids), axis=-1)
    
    def _roi_intensities(self, roi_rect: RoiRectangle, images: npt.NDArray):
        roi_images = roi_rect.slice(images)

        return roi_images.mean(axis=(1, 2))

    def analysis_by_rois(self, named_roi_rects: list[str, RoiRectangle]) -> pd.DataFrame:
        data_frames = []

        for name, roi_rect in named_roi_rects:
            pon_com = self._roi_center_of_masses(roi_rect, self.pon_images)
            poff_com = self._roi_center_of_masses(roi_rect, self.poff_images)
            pon_intensity = self._roi_intensities(roi_rect, self.pon_images)
            poff_intensity = self._roi_intensities(roi_rect, self.poff_images)

            roi_df = pd.DataFrame(data={
                "pon_com_x": pon_com.T[0],
                "pon_com_y": pon_com.T[1],
                "poff_com_x": poff_com.T[0],
                "poff_com_y": poff_com.T[1],
                "pon_intensity": pon_intensity,
                "poff_intensity": poff_intensity
            })
            
            roi_df = roi_df.transpose()

            roi_df.index=[[name]*len(roi_df.index), roi_df.index]
            
            data_frames.append(roi_df)
            
        data_df = pd.concat(data_frames)
        data_df = data_df.transpose()
        data_df.index = self.delay
        
        return data_df
    
if __name__ == "__main__":
    # file = "D:\\dev\\p_python\\xrd\\xfel_sample_data\\Npz_files\\run=0001_scan=0001.npz"
    file = "D:\\dev\\p_python\\xrd\\xfel_sample_data\\Npz_files\\run=062\\scan=001\\run=062, scan=001.npz"
    rd = MeanDataProcessor(file)
    roi_rects = [RoiRectangle(0, 0, None, None), RoiRectangle(100, 200, 500, 600)]
    names = ["total", "center"]
    named_roi_rects = zip(names, roi_rects)
    data_df = rd.analysis_by_rois(named_roi_rects)
    
    import matplotlib.pyplot as plt
    name = 'center'
    plt.plot(data_df.index, data_df[name]["poff_intensity"])
    plt.plot(data_df.index, data_df[name]["pon_intensity"])
    plt.show()
