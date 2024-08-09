from functools import partial

from roi_rectangle import RoiRectangle
from preprocess.generic_preprocessors import (
    div_images_by_qbpm,
    filter_images_qbpm_by_linear_model,
    subtract_dark,
    ransac_regression,
    equalize_brightness,
    add_bias
)

from typing import Callable
import numpy.typing as npt


ImagesQbpmProcessor = Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]


def shift_to_positive(
    images: npt.NDArray,
    qbpm: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Shift the images to ensure all values are non-negative by adding a bias.

    This function adds a bias to the images to ensure that all pixel values are non-negative.
    The bias is calculated as the absolute value of the minimum pixel value in the images.

    Parameters:
    - images (Images): The input images to be shifted.
    - qbpm (Qbpm): The QBPM values corresponding to the images.

    Returns:
    - tuple[Images, Qbpm]: A tuple containing the shifted images and the original QBPM values.
    """
    return add_bias(images), qbpm


def subtract_dark_background(
    images: npt.NDArray,
    qbpm: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Remove the dark background from the images.

    Parameters:
    - images: Images, the input images.
    - qbpm: Qbpm, the Qbpm values.

    Returns:
    - tuple[Images, Qbpm]: The images with dark background removed and the original Qbpm values.
    """

    return subtract_dark(images), qbpm


def normalize_images_by_qbpm(
    images: npt.NDArray,
    qbpm: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Normalize the images by the Qbpm values.

    Parameters:
    - images: Images, the input images.
    - qbpm: Qbpm, the Qbpm values.

    Returns:
    - tuple[Images, Qbpm]: The normalized images and the original Qbpm values.
    """
    return div_images_by_qbpm(images, qbpm), qbpm


def remove_outliers_using_ransac(
    images: npt.NDArray,
    qbpm: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Remove outliers from the images and Qbpm values using RANSAC regression.

    Parameters:
    - images: Images, the input images.
    - qbpm: Qbpm, the Qbpm values.

    Returns:
    - tuple[Images, Qbpm]: The images and Qbpm values with outliers removed.
    """
    mask = ransac_regression(images.sum(axis=(1, 2)), qbpm, min_samples=2)[0]
    return images[mask], qbpm[mask]


def create_ransac_roi_outlier_remover(roi_rect: RoiRectangle) -> ImagesQbpmProcessor:
    """
    Create a function to remove outliers using a linear model with a given sigma.

    Parameters:
    - sigma: float, the sigma value for the outlier removal.

    Returns:
    - ImageQbpmProcessor: A function that takes images and Qbpm values and returns the filtered images and Qbpm values.
    """
    def remove_ransac_roi_outliers(images: npt.NDArray, qbpm: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        roi_image = roi_rect.slice(images)
        mask = ransac_regression(roi_image.sum(axis=(1, 2)), qbpm, min_samples=2)[0]
        return images[mask], qbpm[mask]
    return remove_ransac_roi_outliers


def equalize_intensities(
    images: npt.NDArray,
    qbpm: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Equalize the intensities of the images while keeping the Qbpm values unchanged.

    Parameters:
    - images: np.ndarray, 3D array of images (num_images, height, width).
    - qbpm: np.ndarray, 1D array of Qbpm values corresponding to each image.

    Returns:
    - tuple[np.ndarray, np.ndarray]: A tuple containing the brightness-equalized images and the original Qbpm values.
    """
    return equalize_brightness(images), qbpm


def create_linear_model_outlier_remover(sigma) -> ImagesQbpmProcessor:
    """
    Create a function to remove outliers using a linear model with a given sigma.

    Parameters:
    - sigma: float, the sigma value for the outlier removal.

    Returns:
    - ImageQbpmProcessor: A function that takes images and Qbpm values and returns the filtered images and Qbpm values.
    """
    remove_outlier: ImagesQbpmProcessor = partial(filter_images_qbpm_by_linear_model, sigma=sigma)
    return remove_outlier


def apply_pipeline(
    pipeline: list[ImagesQbpmProcessor],
    images: npt.NDArray,
    qbpm: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Apply a series of image and Qbpm processing functions to the input data.

    Parameters:
    - pipeline: list[ImagesQbpmProcessor], a list of processing functions to apply.
    - images: Images, the input images.
    - qbpm: Qbpm, the Qbpm values.

    Returns:
    - tuple[Images, Qbpm]: The processed images and Qbpm values.
    """
    for processor in pipeline:
        images, qbpm = processor(images, qbpm)
    return images, qbpm
