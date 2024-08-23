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
    from src.processor.saver import MatSaverStrategy, NpzSaverStrategy
    config = load_config()
    npz_file = os.path.join(config.path.npz_dir, "run=0065_scan=0001.npz")

    darks = NpzLoader(npz_file).data['poff']

    print(darks.shape)
    print(darks.max())
    print(darks.min())
    print(darks.mean())
    print(darks.dtype)

    dark_dir = os.path.join(config.path.analysis_dir, 'DARK', 'dark.npy')
    np.save(dark_dir, darks)