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
    npz_file = os.path.join(config.path.npz_dir, "run=0144_scan=0001.npz")
    data: dict = NpzLoader(npz_file).data
    print(list(data.keys()))
