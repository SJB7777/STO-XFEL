import os
from typing import Any

import numpy as np
import pandas as pd
import h5py

from src.utils.file_util import get_file_list, get_folder_list


def get_file_status(root: str) -> dict:

    status: dict = {}

    runs: list[str] = get_folder_list(root)
    for run in runs:
        path = os.path.join(root, run)
        scans = get_folder_list(path)
        for scan in scans:
            path = os.path.join(path, scan)
            files = get_file_list(path)

            name = "_".join(path[-2:])[:-2]

            nums = {int(file[1:-3]) for file in files}
            max_num = max(nums)
            missing_nums = set(range(1, max_num + 1)) - nums
            status[name] = [max_num, missing_nums]

    return status


def h5_tree(val: Any, pre: None = '') -> None:
    """
    with h5py.File(file) as hf:
        print(hf)
        h5_tree(hf)
    """
    items_cnt = len(val)
    for key, val in val.items():
        items_cnt -= 1
        if items_cnt == 0:
            # the last item
            if isinstance(val, h5py._hl.group.Group):
                print(f"{pre}└── {key}")
                h5_tree(val, f'{pre}    ')
            else:
                try:
                    if h5py.check_string_dtype(val.dtype):
                        print(f"{pre}└── {key} ({val})")
                    else:
                        print(f"{pre}└── {key} ({val.shape})")
                except TypeError:
                    print(f"{pre}└── {key} (scalar)")
        else:
            if isinstance(val, h5py._hl.group.Group):
                print(f"{pre}├── {key}")
                h5_tree(val, f"{pre}│   ")
            else:
                try:
                    if h5py.check_string_dtype(val.dtype):
                        print(f"{pre}├── {key} ({val})")
                    else:
                        print(f"{pre}├── {key} ({val.shape})")
                except TypeError:
                    print(f"{pre}├── {key} (scalar)")


def load_matdata(h5file: str) -> pd.DataFrame:
    return pd.read_hdf(h5file, 'metadata')


if __name__ == "__main__":

    from src.utils.file_util import get_run_scan_directory
    from src.config.config import load_config

    config = load_config()
    load_dir = config.path.load_dir

    file = get_run_scan_directory(load_dir, 152, 1, 81)

    with h5py.File(file) as hf:
        print(hf)
        h5_tree(hf)
