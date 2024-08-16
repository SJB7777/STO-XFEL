import numpy as np
from scipy.io import loadmat
import numpy.typing as npt


class MatLoader:
    def __init__(self, file):
        mat_images: npt.NDArray = loadmat(file)["data"]
        images = mat_images.swapaxes(0, 2)
        self.images = images.swapaxes(1, 2)

class NpzLoader:
    def __init__(self, file: str):
        self.data =  dict(np.load(file))


if __name__ == "__main__":
    import os
    from src.config.config import load_config
    config = load_config()
    mat_dir = config.path.mat_dir
    file_name = "run=0143_scan=0001_poff"
    mat_file = os.path.join(mat_dir, file_name + ".mat")
    mat_loader = MatLoader(mat_file)
    off_images = mat_loader.images

    file_name = "run=0143_scan=0001_pon"
    mat_file = os.path.join(mat_dir, file_name + ".mat")
    mat_loader = MatLoader(mat_file)
    on_images = mat_loader.images

    print(f"{off_images.shape = }")
    print(f"{on_images.shape = }")
