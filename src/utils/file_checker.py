import os

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

if __name__ == "__main__":
    from rocking.rocking_scan import ReadRockingH5

    config = load_palxfel_config("config.ini")
    file29 = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=156\\scan=001\\p0029.h5"
    # with h5py.File(file29) as hf:
    #     print(hf)
    #     h5_tree(hf)

    file30 = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=156\\scan=001\\p0030.h5"
    # with h5py.File(file30) as hf:
    #     print(hf)
    #     h5_tree(hf)

    file53 = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=156\\scan=001\\p0053.h5"
    # with h5py.File(file53) as hf:
    #     print(hf)
    #     h5_tree(hf)
    rr = ReadRockingH5(file30)
