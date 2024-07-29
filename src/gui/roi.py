import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from core.loader_strategy import HDF5FileLoader
from cuptlib_config.palxfel import load_palxfel_config
from roi_rectangle import RoiRectangle
from utils.file_util import get_run_scan_directory, get_file_list


drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
rect = None
ax = None

def on_mouse_press(event):
    global ix, iy, fx, fy, drawing, rect, ax

    if event.inaxes is not None:
        if event.button == 1:
            drawing = True
            ix, iy = int(event.xdata), int(event.ydata)
            fx, fy = ix, iy
            if rect is not None:
                rect.remove()
            rect = patches.Rectangle((ix, iy), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.draw()

def on_mouse_release(event):
    global fx, fy, drawing, rect

    if event.inaxes is not None and drawing:
        drawing = False
        fx, fy = int(event.xdata), int(event.ydata)
        if rect is not None:
            rect.set_width(fx - ix)
            rect.set_height(fy - iy)
            plt.draw()

def on_mouse_move(event):
    global fx, fy, drawing, rect

    if event.inaxes is not None and drawing:
        fx, fy = int(event.xdata), int(event.ydata)
        if rect is not None:
            rect.set_width(fx - ix)
            rect.set_height(fy - iy)
            plt.draw()

def select_roi(image):
    global ix, iy, fx, fy, rect, ax

    img = image.copy()

    fig, ax = plt.subplots()
    ax.imshow(img)

    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    plt.show()

    if ix == -1 or iy == -1 or fx == -1 or fy == -1:
        return None
    else:
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        return (x1, y1, x2, y2)

def select_roi_by_run_scan(run: int, scan: int) -> RoiRectangle:
    config = load_palxfel_config("config.ini")
    load_dir = config.path.load_dir
    scan_dir = get_run_scan_directory(load_dir, run, scan)
    files = get_file_list(scan_dir)
    file = os.path.join(scan_dir, files[len(files) // 2])
    hfl = HDF5FileLoader(file)
    image = np.log1p(hfl.images.sum(axis=0))
    
    roi_tuple = select_roi(image)
    return RoiRectangle(*roi_tuple)

if __name__ == "__main__":
    
    file: str = "D:\\dev\\p_python\\xrd\\xfel_sample_data\\run=001\\scan=001\\p0110.h5"
    hfl = HDF5FileLoader(file)
    image = np.log1p(hfl.images.sum(axis=0))
    if image is None:
        print("이미지를 불러올 수 없습니다.")
    else:
        roi_coords = select_roi(image)
        if roi_coords is not None:
            print(f"선택된 ROI 좌표: {roi_coords}")
        else:
            print("ROI가 선택되지 않았습니다.")