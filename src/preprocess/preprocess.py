import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.io import loadmat
from utils.file_util import load_palxfel_config

def get_linear_regression_confidence_lower_upper_bound(
    y: np.ndarray, 
    x: np.ndarray, 
    sigma: float = 3.0
) -> np.ndarray:
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
    y (np.ndarray): Dependent variable data. Shape: (N,)
    x (np.ndarray): Independent variable data. Shape: (N,)
    sigma (float): Number of standard deviations for the confidence interval. Default is 3.0.

    Returns:
    lowerbound (np.ndarray): Lower bounds of y
    upperbound (np.ndarray): Upper bounds of y
    y_fit (np.ndarray): Fitted y

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
    images: np.ndarray, qbpm: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, ...]:
    """
    Filter images based on the confidence interval of their intensities using a linear regression model with QBPM values.

    This function computes the total intensity of each image, applies a linear regression model 
    to the intensity and QBPM values, and generates a mask to filter out images whose intensities 
    fall outside the specified confidence interval.

    Parameters:
    images (np.ndarray): Array of images. Shape: (N, H, W), where N is the number of images, and H and W are the height and width of each image.
    qbpm (np.ndarray): Array of QBPM (Quadrature Balanced Photodetector Measurements) values. Shape: (N,)
    sigma (float): Number of standard deviations for the confidence interval.

    Returns:
    tuple[np.ndarray, ...]: Tuple of filtered intensities and QBPM values.
        - Filtered intensities (np.ndarray): Array of intensities within the confidence interval. Shape: (M,), where M <= N.
        - Filtered qbpm (np.ndarray): Array of QBPM values corresponding to the filtered intensities. Shape: (M,), where M <= N.
    
    Note:
    This method uses the `get_linear_regression_confidence_lower_upper_bound` function to generate the mask based 
    on the linear regression model and confidence interval.
    """
    intensites = images.sum(axis=(1, 2))
    lower_bound, upper_bound, _ = get_linear_regression_confidence_lower_upper_bound(intensites, qbpm, sigma)
    mask = np.logical_and(intensites >= lower_bound, intensites <= upper_bound)
    
    return images[mask], qbpm[mask]

def nomalize_by_qbpm(images, qbpm) -> np.ndarray:
    """
    Divide images by qbpm.
    
    Parameters:
    images (np.ndarray): Array of images. Shape: (N, H, W), where N is the number of images, and H and W are the height and width of each image.
    qbpm (np.ndarray): Array of QBPM (Quadrature Balanced Photodetector Measurements) values. Shape: (N,)
    
    Returns:
    np.ndarray: Images that divided by qbpm
    """
    return images / qbpm[:, np.newaxis, np.newaxis]

def subtract_dark(images: np.ndarray) -> np.ndarray:
    config = load_palxfel_config("config.ini")
    dark_file = os.path.join(config.path.save_dir, "DARK\\dark.npy")
    dark_images = np.load(dark_file)
    dark = np.mean(dark_images, axis=0)
    return np.maximum(images - dark[np.newaxis, :, :], 0)

if __name__ == "__main__":
    sample = np.random.randint(0, 255, (10, 514, 1030))
    result = subtract_dark(sample)
    print(result.shape)
    print(result)