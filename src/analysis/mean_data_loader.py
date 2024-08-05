from scipy.io import loadmat

import numpy.typing as npt

class MatLoader:
    def __init__(self, file):
        mat_images:npt.NDArray = loadmat(file)["data"]
        images = mat_images.swapaxes(0, 2)
        self.images = images.swapaxes(1, 2)

if __name__ == "__main__":
    import os
    from core.saver import TifSaverStrategy
    from config import load_config
    config = load_config()
    mat_dir = config.path.mat_dir
    file_name = "run=0001_scan=0001_empty_poff"
    mat_file = os.path.join(mat_dir, file_name + ".mat")
    mat_loader = MatLoader(mat_file)
    images = mat_loader.images


    tif_dir = config.path.tif_dir
    tif_file = os.path.join(tif_dir, file_name + ".tif")

    tif_saver = TifSaverStrategy()
    tif_saver.save(file_name, {"data": images})