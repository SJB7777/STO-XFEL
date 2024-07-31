from scipy.io import loadmat

import numpy.typing as npt

class MatLoader:
    def __init__(self, file):
        mat_images:npt.NDArray = loadmat(file)["data"]
        images = mat_images.swapaxes(0, 2)
        self.images = images.swapaxes(1, 2)


if __name__ == "__main__":
    from core.saver import TifSaverStrategy
    from preprocess.remove_continuous_noise import remove_noise
    mat_file = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\Mat_files2\\run=0176_scan=0001_no_normalize_poff.mat"
    mat_loader = MatLoader(mat_file)
    original_images = mat_loader.images

    denoised_images:npt.NDArray = remove_noise(original_images, 0.1)


    tif_saver = TifSaverStrategy()
    tif_saver.save("no_normalize_poff", {"data": original_images})
    tif_saver.save("denoised_no_normalize_poff", {"data": denoised_images})