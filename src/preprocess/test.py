import numpy as np

import numpy.typing as npt

def remove_noise(images: npt.NDArray, threshold: float, front: int=5, back :int=5):
    N = 15
    n_th = N*0.6

    images_front = images[:front]
    array_sel_th = images_front >= threshold
    noise_addr = np.prod(array_sel_th, axis=0)
    noise_avg = np.mean(images_front*noise_addr, axis=0)
    array_sel_noise_removal = images - noise_avg

    array_sel_2 = array_sel_noise_removal[-back:]
    array_sel_th_2 = array_sel_2 >= threshold
    noise_addr_2 = np.sum(array_sel_th_2, axis=0) >= n_th
    
    noise_avg_2 = np.mean(array_sel_2 * noise_addr_2, axis=0)
    array_sel_noise_removal_2 = array_sel_noise_removal - noise_avg_2
    array_sel_noise_removal = np.abs(array_sel_noise_removal_2)

    return array_sel_noise_removal

# 예제 사용법
if __name__ == "__main__":
    import os
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    from roi_rectangle import RoiRectangle

    from cuptlib_config.palxfel import load_palxfel_config
    from core.raw_data_processor import HDF5FileLoader

    file = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=177\\scan=001\\p0041.h5"
    # images = loadmat("Y:\\240608_FXS\\raw_data\\h5\\type=raw\\mat_files\\run=0186_scan=0001.mat")["data"]
    # images = images.swapaxes(0, 2)
    # images = images.swapaxes(1, 2)
    # image = images.mean(axis=0)
    rr = HDF5FileLoader(file)
    images = rr.images
    qbpm = rr.qbpm_sum
    # images = images / qbpm[:, np.newaxis, np.newaxis]
    image = images.mean(axis=0)
    # clean_image_array = remove_noise(images)

    # plt.imshow(np.log1p(images[40]))
    # plt.show()

    config = load_palxfel_config("config.ini")
    dark_file = os.path.join(config.path.save_dir, "DARK\\dark.npy")
    dark_images = np.load(dark_file)

    
    print("image:", image.min(), image.max())

    # roi_rect = RoiRectangle(0, 0, image.shape[0], image.shape[1])
    # dark_images = roi_rect.slice(dark_images)
    dark_image = dark_images.mean(axis=0)

    
    # dark_image = np.where(dark_image < 0.5, 0, dark_image)
    # image = np.where(image < 0.5, 0, image)
    print("dark:", dark_image.min(), dark_image.max())
        # 히스토그램 계산
    hist_image, bins_image = np.histogram(np.log1p(image.flatten()), bins=300, density=True)
    hist_dark, bins_dark = np.histogram(np.log1p(dark_image.flatten()), bins=300, density=True)

    # # 가장 높은 빈도 제외
    # hist_image[np.argmax(hist_image)] = 0
    # hist_dark[np.argmax(hist_dark)] = 0

    # 그래프 그리기
    fig, ax = plt.subplots(1, 1)
    ax.step(bins_image[:-1][:50], hist_image[:50], where='post', label='image')
    ax.step(bins_dark[:-1][:50], hist_dark[:50], where='post', label='dark_image')
    ax.set_title("image")
    ax.legend()

    plt.show()

    