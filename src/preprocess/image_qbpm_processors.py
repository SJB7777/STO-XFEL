from functools import partial

from roi_rectangle import RoiRectangle
import numpy.typing as npt
from typing import Callable, Tuple

from preprocess.preprocessing_functions import (
    normalize_by_qbpm, 
    filter_images_qbpm_by_linear_model,
    subtract_dark, 
    RANSAC_regression,
    equalize_brightness
    )

Images = npt.NDArray
Qbpm = npt.NDArray
ImagesQbpmProcessor = Callable[[Images, Qbpm], Tuple[Images, Qbpm]]

def subtract_dark_background(images: Images, qbpm: Qbpm) -> Tuple[Images, Qbpm]:
    """
    Remove the dark background from the images.

    Parameters:
    - images: Images, the input images.
    - qbpm: Qbpm, the Qbpm values.

    Returns:
    - Tuple[Images, Qbpm]: The images with dark background removed and the original Qbpm values.
    """
    return subtract_dark(images), qbpm

def normalize_images_by_qbpm(images: Images, qbpm: Qbpm) -> Tuple[Images, Qbpm]:
    """
    Normalize the images by the Qbpm values.

    Parameters:
    - images: Images, the input images.
    - qbpm: Qbpm, the Qbpm values.

    Returns:
    - Tuple[Images, Qbpm]: The normalized images and the original Qbpm values.
    """
    return normalize_by_qbpm(images, qbpm), qbpm

def remove_by_ransac(images: Images, qbpm: Qbpm) -> Tuple[Images, Qbpm]:
    """
    Remove outliers from the images and Qbpm values using RANSAC regression.

    Parameters:
    - images: Images, the input images.
    - qbpm: Qbpm, the Qbpm values.

    Returns:
    - Tuple[Images, Qbpm]: The images and Qbpm values with outliers removed.
    """
    mask = RANSAC_regression(images.sum(axis=(1, 2)), qbpm, min_samples=2)[0]
    return images[mask], qbpm[mask]

def create_remove_by_ransac_roi(roi_rect: RoiRectangle) -> ImagesQbpmProcessor:
    def remove_by_ransac_roi(images: Images, qbpm: Qbpm) -> Tuple[Images, Qbpm]:
        roi_image = roi_rect.slice(images)
        mask = RANSAC_regression(roi_image.sum(axis=(1, 2)), qbpm, min_samples=2)[0]
        return images[mask], qbpm[mask]
    return remove_by_ransac_roi

def equalize_intensities(images: Images, qbpm: Qbpm) -> Tuple[Images, Qbpm]:
    """
    Equalize the intensities of the images while keeping the Qbpm values unchanged.

    Parameters:
    - images: np.ndarray, 3D array of images (num_images, height, width).
    - qbpm: np.ndarray, 1D array of Qbpm values corresponding to each image.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the brightness-equalized images and the original Qbpm values.
    """
    return equalize_brightness(images), qbpm

def create_outlier_remover(sigma) -> ImagesQbpmProcessor:
    """
    Create a function to remove outliers using a linear model with a given sigma.

    Parameters:
    - sigma: float, the sigma value for the outlier removal.

    Returns:
    - ImageQbpmProcessor: A function that takes images and Qbpm values and returns the filtered images and Qbpm values.
    """
    remove_outlier: ImagesQbpmProcessor = partial(filter_images_qbpm_by_linear_model, sigma=sigma)
    return remove_outlier

