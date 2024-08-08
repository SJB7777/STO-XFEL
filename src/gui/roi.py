import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from core.loader_strategy import HDF5FileLoader
from roi_rectangle import RoiRectangle
from utils.file_util import get_run_scan_directory, get_file_list
from config import load_config

from typing import Optional, Union
import numpy.typing as npt

class RoiSelector:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        self.rect = None
        self.ax = None

    def on_mouse_press(self, event):

        if event.inaxes is not None:
            if event.button == 1:
                self.drawing = True
                self.ix, self.iy = int(event.xdata), int(event.ydata)
                self.fx, self.fy = self.ix, self.iy
                if self.rect is not None:
                    self.rect.remove()
                self.rect = patches.Rectangle((self.ix, self.iy), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                self.ax.add_patch(self.rect)
                plt.draw()

    def on_mouse_release(self, event):

        if event.inaxes is not None and self.drawing:
            self.drawing = False
            self.fx, self.fy = int(event.xdata), int(event.ydata)
            if self.rect is not None:
                self.rect.set_width(self.fx - self.ix)
                self.rect.set_height(self.fy - self.iy)
                plt.draw()

    def on_mouse_move(self, event):

        if event.inaxes is not None and self.drawing:
            self.fx, self.fy = int(event.xdata), int(event.ydata)
            if self.rect is not None:
                self.rect.set_width(self.fx - self.ix)
                self.rect.set_height(self.fy - self.iy)
                plt.draw()

    def select_roi(self, image: npt.NDArray):

        fig, self.ax = plt.subplots()
        self.ax.imshow(image)

        fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        plt.show()

        if self.ix == -1 or self.iy == -1 or self.fx == -1 or self.fy == -1:
            return None
        else:
            x1, y1 = min(self.ix, self.fx), min(self.iy, self.fy)
            x2, y2 = max(self.ix, self.fx), max(self.iy, self.fy)
            return (x1, y1, x2, y2)


def select_roi_by_run_scan(run: int, scan: int, index_mode: Union[int, str]="auto") -> Optional[RoiRectangle]:
    config = load_config()
    load_dir = config.path.load_dir
    scan_dir = get_run_scan_directory(load_dir, run, scan)
    files = get_file_list(scan_dir)

    if index_mode == "auto":
        index = len(files) // 2
    elif isinstance(index_mode, int):
        index = index_mode

    file = os.path.join(scan_dir, files[index])
    hfl = HDF5FileLoader(file)
    data = hfl.get_data()
    images = data.get("poff", None)
    if images is None:
        images = data.get("pon", None)
        print("no off data")

    image = np.log1p(images.sum(axis=0))
    
    roi_tuple = RoiSelector().select_roi(image)
    if roi_tuple is None:
        return None
    return RoiRectangle(*roi_tuple)


if __name__ == "__main__":
    from config import load_config
    from utils.file_util import get_run_scan_directory

    config = load_config()
    load_dir = config.path.load_dir
    file = get_run_scan_directory(load_dir, 154, 1, 1)
    loader = HDF5FileLoader(file)

    image = np.log1p(loader.images.sum(axis=0))
    roi = RoiSelector().select_roi(image)
    
    print(roi)