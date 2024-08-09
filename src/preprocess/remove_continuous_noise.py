import numpy as np

import numpy.typing as npt


def remove_noise(images: npt.NDArray, threshold: float, front: int = 5, back: int = 5):
    N = 15
    n_th = N * 0.6

    images_front = images[:front]
    array_sel_th = images_front >= threshold
    noise_addr = np.prod(array_sel_th, axis=0)
    noise_avg = np.mean(images_front * noise_addr, axis=0)
    array_sel_noise_removal = images - noise_avg

    array_sel_2 = array_sel_noise_removal[-back:]
    array_sel_th_2 = array_sel_2 >= threshold
    noise_addr_2 = np.sum(array_sel_th_2, axis=0) >= n_th

    noise_avg_2 = np.mean(array_sel_2 * noise_addr_2, axis=0)
    array_sel_noise_removal_2 = array_sel_noise_removal - noise_avg_2
    array_sel_noise_removal = np.abs(array_sel_noise_removal_2)

    return array_sel_noise_removal


if __name__ == "__main__":
    from scipy.io import loadmat, savemat
    from roi_rectangle import RoiRectangle

    file = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\Mat_files2\\run=0176_scan=0001_no_normalize_poff.mat"
    images = loadmat(file)["data"]
    images = images.swapaxes(0, 2)
    images = images.swapaxes(1, 2)

    denoised_images = remove_noise(images, 5)

    savemat("denoised_images.mat")

    roi_rect = RoiRectangle(0, 0, 500, 500)
    roi_rect.slice(images)
