import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.io import loadmat
from sklearn.linear_model import RANSACRegressor

from utils.file_util import load_palxfel_config

import numpy.typing as npt
from typing import Optional, Callable

def RANSAC_regression(y: np.ndarray, x: np.ndarray, min_samples: Optional[int] = None) -> tuple[npt.NDArray[np.bool_], npt.NDArray, npt.NDArray]:
    """
    Perform RANSAC (Random Sample Consensus) regression to identify inliers and estimate the regression model.

    RANSAC is an iterative method to estimate parameters of a mathematical model from a set of observed data
    that contains outliers. This algorithm is non-deterministic and aims to separate the training data into
    inliers (data points that are subject to noise) and outliers (data points that do not fit the model).

    Parameters:
    - y (NDArray): The target variable array.
    - x (NDArray): The feature variable array.
    - min_samples (int, optional): The minimum number of samples to fit the model. Default is 3.

    Returns:
    - inlier_mask (NDArray): A boolean array where True indicates an inlier and False indicates an outlier.
    - coef (float): The coefficient of the linear model.
    - intercept (float): The intercept of the linear model.

    Example:
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> inlier_mask, coef, intercept = RANSAC_regression(y, x)
    >>> print(inlier_mask)
    [ True  True  True  True  True]
    >>> print(coef)
    [1.]
    >>> print(intercept)
    0.0
    """
    X = x[:, np.newaxis]
    ransac = RANSACRegressor(min_samples=min_samples).fit(X, y)
    inlier_mask = ransac.inlier_mask_
    return inlier_mask, ransac.estimator_.coef_, ransac.estimator_.intercept_

def get_linear_regression_confidence_lower_upper_bound(
    y: npt.NDArray, 
    x: npt.NDArray, 
    sigma: float
) -> npt.NDArray:
    """
    Get lowerbound and upperbound for data points based on their confidence interval in a linear regression model.

    Statistical Explanation:
    This function applies the principle of propagation of uncertainty. It fits a linear 
    regression model y = mx + b, calculates the standard errors of the model parameters 
    m and b, and uses these to generate prediction intervals for identifying outliers.

    The prediction interval is calculated using the formula:
    y ± sqrt((m_err * x)^2 + b_err^2) * sigma
    where m_err and b_err are the standard errors of m and b respectively.

    Parameters:
    y (NDArray): Dependent variable data. Shape: (N,)
    x (NDArray): Independent variable data. Shape: (N,)
    sigma (float): Number of standard deviations for the confidence interval. Default is 3.0.

    Returns:
    lowerbound (NDArray): Lower bounds of y
    upperbound (NDArray): Upper bounds of y
    y_fit (NDArray): Fitted y

    Note:
    This method assumes a linear relationship in the data. For strong non-linearities,
    a different approach may be necessary.
    """
    def linear_model(x, m, b):
        return m * x + b
    
    params, covars = curve_fit(linear_model, x, y)

    m, b = params
    m_err, b_err = np.sqrt(np.diag(covars))
    y_fit = linear_model(x, m, b)
    
    # Calculate upper and lower bounds considering both slope and intercept errors
    error = np.sqrt((m_err * x)**2 + b_err**2)
    upper_bound = y_fit + error * sigma
    lower_bound = y_fit - error * sigma

    return lower_bound, upper_bound, y_fit

def filter_images_qbpm_by_linear_model(
    images: npt.NDArray, qbpm: npt.NDArray, sigma: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Filter images based on the confidence interval of their intensities using a linear regression model with QBPM values.

    This function computes the total intensity of each image, applies a linear regression model 
    to the intensity and QBPM values, and generates a mask to filter out images whose intensities 
    fall outside the specified confidence interval.

    Parameters:
    images (NDArray): Array of images. Shape: (N, H, W), where N is the number of images, and H and W are the height and width of each image.
    qbpm (NDArray): Array of QBPM (Quadrature Balanced Photodetector Measurements) values. Shape: (N,)
    sigma (float): Number of standard deviations for the confidence interval.

    Returns:
    tuple[NDArray, NDArray]: Tuple of filtered intensities and QBPM values.
        - Filtered intensities (NDArray): Array of intensities within the confidence interval. Shape: (M,), where M <= N.
        - Filtered qbpm (NDArray): Array of QBPM values corresponding to the filtered intensities. Shape: (M,), where M <= N.
    
    Note:
    This method uses the `get_linear_regression_confidence_lower_upper_bound` function to generate the mask based 
    on the linear regression model and confidence interval.
    """
    intensites = images.sum(axis=(1, 2))
    lower_bound, upper_bound, _ = get_linear_regression_confidence_lower_upper_bound(intensites, qbpm, sigma)
    mask = np.logical_and(intensites >= lower_bound, intensites <= upper_bound)
    
    return images[mask], qbpm[mask]

def normalize_by_qbpm(images: npt.NDArray, qbpm: npt.NDArray) -> npt.NDArray:
    """
    Divide images by qbpm.
    
    Parameters:
    images (NDArray): Array of images. Shape: (N, H, W), where N is the number of images, and H and W are the height and width of each image.
    qbpm (NDArray): Array of QBPM (Quadrature Balanced Photodetector Measurements) values. Shape: (N,)
    
    Returns:
    NDArray: Images that divided by qbpm
    """
    return images * qbpm.mean() / qbpm[:, np.newaxis, np.newaxis]

def subtract_dark(images: npt.NDArray) -> npt.NDArray:
    config = load_palxfel_config("config.ini")
    dark_file = os.path.join(config.path.save_dir, "DARK\\dark.npy")
    dark_images = np.load(dark_file)
    dark = np.mean(dark_images, axis=0)
    # return np.maximum(images - dark[np.newaxis, :, :], 0)
    return images - dark[np.newaxis, :, :]

def equalize_brightness(images: np.ndarray) -> np.ndarray:
    """
    Equalize the brightness of each image in the 3D array while maintaining the overall average brightness.

    Parameters:
    - images: np.ndarray, 3D array of images (num_images, height, width).

    Returns:
    - np.ndarray: The brightness-equalized images.
    """
    intensites = images.sum(axis=(1, 2))
    overal_normed_intensites = intensites / intensites.mean()
    equalized_images = images / overal_normed_intensites[:, np.newaxis, np.newaxis]
    
    return equalized_images


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from core.loader_strategy import HDF5FileLoader
    from sklearn.linear_model import RANSACRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    from tqdm import tqdm
    
    file: str = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=176\\scan=001\\p0044.h5"

    rr = HDF5FileLoader(file)
    images = rr.images
    qbpm = rr.qbpm_sum
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    intensities = images.sum(axis=(1, 2))
    sorted_indices = np.argsort(qbpm)
    axs[0].scatter(qbpm[sorted_indices], intensities[sorted_indices])
    axs[1].scatter(qbpm[sorted_indices], (intensities / qbpm)[sorted_indices])

    axs[0].set_xlim(0, None)
    axs[0].set_ylim(0, None)
    axs[1].set_xlim(0, None)
    axs[1].set_ylim(0, None)
    plt.show()
 

    intensities = images.sum(axis=(1, 2))
    intensities = intensities / intensities.mean()
    images_div_by_intensities = images / intensities[:, np.newaxis, np.newaxis]
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(np.log1p(images.sum(axis=0)))
    axs[0, 1].imshow(np.log1p(images_div_by_intensities.sum(axis=0)))
    axs[1, 0].scatter(qbpm, images.sum(axis=(1, 2)))
    axs[1, 1].scatter(qbpm, images_div_by_intensities.sum(axis=(1, 2)))
    
    axs[1, 0].set_ylim(0, None)
    axs[1, 1].set_ylim(0, axs[1, 0].get_ylim()[1])
    plt.tight_layout()
    plt.show()


    mask = RANSAC_regression(images.sum(axis=(1, 2)), qbpm, min_samples=2)[0]
    good_images = images[mask]
    good_intensites = good_images.sum(axis=(1, 2))
    good_intensites = good_intensites / good_intensites.mean()
    normed_images = good_images / good_intensites[:, np.newaxis, np.newaxis]
    
    # plt.imshow(np.log1p(normed_images.sum(axis=0)))
    plt.scatter(range(len(good_intensites)), good_intensites)
    plt.show()
    quit()
    min_samples_list = range(1, 11)
    residuals_list = []
    r2_scores = []
    rmse_scores = []

    fig, axs = plt.subplots(2, 5, figsize=(18, 9))
    axs = axs.flatten()

    for i, min_samples in tqdm(enumerate(min_samples_list), total=len(min_samples_list)):
        ransac = RANSACRegressor(min_samples=min_samples)
        ransac.fit(X, y)
        y_pred = ransac.predict(X)
        mask = ransac.inlier_mask_
        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_

        def lin(x):
            return coef[0] * x + intercept

        residuals = y - y_pred
        residuals_list.append(residuals)

        r2 = r2_score(y[mask], y_pred[mask])
        r2_scores.append(r2)

        rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
        rmse_scores.append(rmse)

        axs[i].scatter(X[mask], y[mask], color="blue", label="Inliers")
        axs[i].scatter(X[~mask], y[~mask], color="red", label="Outliers")
        axs[i].plot([X.min(), X.max()], [lin(X.min()), lin(X.max())], color="green")
        axs[i].set_title(f"RANSAC - minsamples={min_samples}")
        axs[i].legend()

        # y축과 x축을 0부터 보이게 설정
        axs[i].set_ylim(0, None)
        axs[i].set_xlim(0, None)

    plt.tight_layout()
    plt.show()

    # 결과 출력
    print("min_samples\tR²\tRMSE")
    for i, min_samples in enumerate(min_samples_list):
        print(f"{min_samples}\t{r2_scores[i]:.4f}\t{rmse_scores[i]:.4f}")

    # 잔차 시각화
    # fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    # axs = axs.flatten()

    # for i, min_samples in enumerate(min_samples_list):
    #     axs[i].hist(residuals_list[i][mask], bins=20, color="blue", alpha=0.7)
    #     axs[i].set_title(f"Residuals - minsamples={min_samples}")

    # plt.tight_layout()
    # plt.show()
