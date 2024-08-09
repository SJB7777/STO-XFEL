import os
import json
from typing import Optional

from src.config import load_config

from roi_rectangle import RoiRectangle
import numpy as np
import numpy.typing as npt
import scipy.io



def get_file_list(mother: str = ".") -> list[str]:
    """
    Get a list of files in the specified directory or the current directory if no directory is specified.

    Args:
        mother (str, optional): The directory path to search for files.
            Defaults to None, which represents the current directory.

    Returns:
        list: A list of filenames in the specified directory.
    """

    files = [file for file in os.listdir(mother) if os.path.isfile(os.path.join(mother, file))]
    return files


def get_folder_list(mother: str = ".") -> list[str]:
    """
    Get a list of folders (directories) in the specified directory or the current directory if no directory is specified.

    Args:
        mother (str, optional): The directory path to search for folders.
            Defaults to None, which represents the current directory.

    Returns:
        list: A list of folder names in the specified directory.
    """
    folders = [folder for folder in os.listdir(mother) if os.path.isdir(os.path.join(mother, folder))]
    return folders


def create_idx_path(mother: str, suffix: str = "") -> str:
    folders = get_folder_list(mother)
    idxes = []
    for folder in folders:
        try:
            idx = int(folder.split("=")[1].split("_")[0])
            idxes.append(idx)
        except (ValueError, IndexError):
            pass
    if idxes:
        idx = max(idxes) + 1
    else:
        idx = 0
    folder_name = f"idx={idx}_{suffix}"
    values_path = os.path.join(mother, folder_name)
    os.makedirs(values_path, exist_ok=True)
    return values_path


def get_run_scan_directory(mother: str, run: int, scan: Optional[int] = None, file_num: Optional[int] = None) -> str:
    """
    Generate the directory for a given run and scan number, optionally with a file number.

    Parameters:
        mother (str): The base directory or path where the path will be generated.
        run (int): The run number for which the path will be generated.
        scan (int, optional): The scan number for which the path will be generated.
            If not provided, only the run directory path will be returned.
        file_num (int, optional): The file number for which the path will be generated.
            If provided, both run and scan directories will be included in the path.

    Returns:
        str: The path representing the specified run, scan, and file number (if applicable).
    """

    if scan is None and file_num is None:
        return os.path.join(mother, f"run={run:0>3}")
    if scan is not None and file_num is None:
        return os.path.join(mother, f"run={run:0>3}", f"scan={scan:0>3}")
    if scan is not None and file_num is not None:
        return os.path.join(mother, f"run={run:0>3}", f"scan={scan:0>3}", f"p{file_num:0>4}.h5")


def create_run_scan_directory(dir: str, run: int, scan: int) -> str:
    """
    Create a nested directory structure for the given run and scan numbers.

    Parameters:
        dir (str): The base directory where the nested structure will be created.
        run (int): The run number for which the directory will be created.
        scan (int): The scan number for which the directory will be created.

    Returns:
        str: The path of the created nested directory.
    """

    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f'run={run:0>3d}')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'scan={scan:0>3d}')
    os.makedirs(path, exist_ok=True)
    return path


def format_run_scan_filename(run: int, scan: Optional[int] = None, file_num: Optional[int] = None) -> str:
    """
    Generate a formatted file name based on the provided run, scan, and file number.

    Parameters:
        run (int): The run number to be included in the file name.
        scan (int, optional): The scan number to be included in the file name.
            If not provided, only the run number will be included.
        file_num (int, optional): The file number to be included in the file name.
            If provided, both run and scan numbers will be included.

    Returns:
        str: The formatted file name containing run, scan, and file numbers (if applicable) separated by underscores.
    """

    if scan is None and file_num is None:
        return f"run={run:0>3}"
    if scan is not None and file_num is None:
        return "_".join([f"run={run:0>3}", f"scan={scan:0>3}"])
    if scan is not None and file_num is not None:
        return "_".join([f"run={run:0>3}", f"scan={scan:0>3}", f"p{file_num:0>4}"])


def get_roi_list(mother: str) -> Optional[list[RoiRectangle]]:

    file_path = os.path.join(mother, 'ROI_coords.json')
    if os.path.exists(file_path):
        # If paramter file exists, open json file.
        with open(file_path, 'r') as f:
            try:
                roi_rect_list = json.load(f)
            except json.decoder.JSONDecodeError:
                roi_rect_list = None
    else:
        roi_rect_list = None

    if not roi_rect_list:
        roi_rect_list = None

    if isinstance(roi_rect_list, list):
        roi_rect_list = [RoiRectangle(*region) for region in roi_rect_list]

    return roi_rect_list


def get_ooi(mother: str) -> Optional[RoiRectangle]:
    file_path = os.path.join(mother, 'OOI_coords.txt')
    if os.path.exists(file_path):
        # If paramter file exists, open json file.
        with open(file_path, 'r') as f:

            temp = list(map(int, f.readline().split()))
            ooi_rect = RoiRectangle(*temp)

    else:
        ooi_rect = None

    return ooi_rect


def get_sigma_factor(mother: str) -> Optional[float]:

    file_path = os.path.join(mother, 'sigma_factor.txt')
    if os.path.exists(file_path):
        with open(os.path.join(mother, 'sigma_factor.txt'), 'r') as f:
            sig_fac = float(f.read())
    else:
        sig_fac = None

    return sig_fac


def save_roi_list(mother: str, roi_rect_list: list[RoiRectangle]) -> None:
    region_list = [list(region.get_coordinate()) for region in roi_rect_list]
    file_name = os.path.join(mother, 'ROI_coords.json')
    with open(file_name, 'w') as f:
        f.write(json.dumps(region_list))


def save_ooi(mother: str, ooi_rect: RoiRectangle) -> None:

    with open(os.path.join(mother, 'OOI_coords.txt'), 'w') as f:
        f.write(f"{ooi_rect.x1} {ooi_rect.y1} {ooi_rect.x2} {ooi_rect.y2}")


def save_sigma_factor(mother: str, sig_fac: float) -> None:
    with open(os.path.join(mother, 'sigma_factor.txt'), 'w') as f:
        f.write(str(sig_fac))


def mat_to_ndarray(run: int, scan: int) -> npt.NDArray:
    config = load_config()
    path = os.path.join(config.path.mat_dir, f'run={run:0>3d}_scan={scan}.mat')
    mat_data = scipy.io.loadmat(path)
    images = mat_data["data"]
    return np.transpose(images, axes=range(images.ndim)[::-1])
