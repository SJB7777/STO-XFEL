from functools import partial, reduce
from typing import Callable

import numpy.typing as npt
from roi_rectangle import RoiRectangle

from src.preprocessor.generic_preprocessors import (
    div_images_by_qbpm,
    filter_images_qbpm_by_linear_model,
    subtract_dark,
    ransac_regression,
    equalize_brightness,
    add_bias
)


ImagesQbpm = tuple[npt.NDArray, npt.NDArray]
ImagesQbpmProcessor = Callable[[ImagesQbpm], ImagesQbpm]


def shift_to_positive(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """
    Shift the images to ensure all values are non-negative by adding a bias.

    This function adds a bias to the images to ensure that all pixel values are non-negative.
    The bias is calculated as the absolute value of the minimum pixel value in the images.

    Parameters:
    - images_qbpm (tuple[Images, Qbpm]): tuple of Images and Qbpm

    Returns:
    - tuple[Images, Qbpm]: A tuple containing the shifted images and the original QBPM values.
    """
    return add_bias(images_qbpm[0]), images_qbpm[1]


def subtract_dark_background(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """
    Remove the dark background from the images.

    Parameters:
    - images_qbpm (tuple[Images, Qbpm]): tuple of Images and Qbpm

    Returns:
    - tuple[Images, Qbpm]: The images with dark background removed and the original Qbpm values.
    """

    return subtract_dark(images_qbpm[0]), images_qbpm[1]


def normalize_images_by_qbpm(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """
    Normalize the images by the Qbpm values.

    Parameters:
    - images_qbpm (tuple[Images, Qbpm]): tuple of Images and Qbpm

    Returns:
    - tuple[Images, Qbpm]: The normalized images and the original Qbpm values.
    """
    return div_images_by_qbpm(images_qbpm[0], images_qbpm[1]), images_qbpm[1]


def remove_outliers_using_ransac(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """
    Remove outliers from the images and Qbpm values using RANSAC regression.

    Parameters:
    - images_qbpm (tuple[Images, Qbpm]): tuple of Images and Qbpm

    Returns:
    - tuple[Images, Qbpm]: The images and Qbpm values with outliers removed.
    """
    mask = ransac_regression(images_qbpm[0].sum(axis=(1, 2)), images_qbpm[1], min_samples=2)[0]
    return images_qbpm[0][mask], images_qbpm[1][mask]


def create_ransac_roi_outlier_remover(roi_rect: RoiRectangle) -> ImagesQbpmProcessor:
    """
    Create a function to remove outliers using a linear model with a given sigma.

    Parameters:
    - sigma: float, the sigma value for the outlier removal.

    Returns:
    - ImageQbpmProcessor: A function that takes ImagesQbpm and returns the filtered ImagesQbpm.
    """
    def remove_ransac_roi_outliers(images_qbpm: ImagesQbpm) -> ImagesQbpm:
        roi_image = roi_rect.slice(images_qbpm[0])
        mask = ransac_regression(roi_image.sum(axis=(1, 2)), images_qbpm[1], min_samples=2)[0]
        return images_qbpm[0][mask], images_qbpm[1][mask]
    return remove_ransac_roi_outliers


def equalize_intensities(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """
    Equalize the intensities of the images while keeping the Qbpm values unchanged.

    Parameters:
    - images_qbpm (tuple[Images, Qbpm]): tuple of Images and Qbpm

    Returns:
    - tuple[np.ndarray, np.ndarray]: A tuple containing the brightness-equalized images and the original Qbpm values.
    """
    return equalize_brightness(images_qbpm[0]), images_qbpm[1]


def create_linear_model_outlier_remover(sigma) -> ImagesQbpmProcessor:
    """
    Create a function to remove outliers using a linear model with a given sigma.

    Parameters:
    - sigma: float, the sigma value for the outlier removal.

    Returns:
    - ImageQbpmProcessor: A function that takes ImagesQbpm and returns the filtered ImagesQbpm.
    """
    remove_outlier: ImagesQbpmProcessor = partial(filter_images_qbpm_by_linear_model, sigma=sigma)
    return remove_outlier


def compose(*funcs: Callable):
    """combine multiple functions"""
    return reduce(lambda f, g: lambda x: f(g(x)), funcs)