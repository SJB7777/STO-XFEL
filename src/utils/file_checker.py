import os

import pandas as pd
import h5py
from cuptlib_config.palxfel import load_palxfel_config

from utils.file_util import get_file_list, get_folder_list

from typing import Any


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

def h5_tree(val: Any, pre: None ='') -> None:
    items_cnt = len(val)
    for key, val in val.items():
        items_cnt -= 1
        if items_cnt == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')

def load_matdata(h5file: str) -> pd.DataFrame:
    return pd.read_hdf(h5file, 'metadata')
    

if __name__ == "__main__":

    file = "D:\\dev\\p_python\\xrd\\xfel_sample_data\\run=001\\scan=001\\p0110.h5"
    # with h5py.File(file53) as hf:
    #     print(hf)
    #     h5_tree(hf)
    # rr = ReadRockingH5(file30)
    
    metadata = load_matdata(file)
    metadata.to_csv("metadata.csv")
