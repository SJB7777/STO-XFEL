import os

import numpy as np
from roi_rectangle import RoiRectangle

from src.gui.roi import RoiSelector
from src.config.config import load_config
from src.analyzer.loader import NpzLoader


def select_roi(run_n: int) -> RoiRectangle:
    config = load_config()
    npz_file = os.path.join(config.path.npz_dir, f"run={run_n:0>4}_scan=0001.npz")
    image = NpzLoader(npz_file).data['poff'].sum(0)
    roi = RoiSelector().select_roi(np.log1p(image))
    return roi



# if __name__ == "__main__":

#     run_n: int = int(input("Enter run number to select roi: "))
#     roi: tuple[int, int, int, int] = select_roi(run_n)

#     print("Selected roi is", roi)