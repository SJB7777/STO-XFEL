import os

from utils.file_util import get_file_list, get_folder_list


def get_file_status(root: str):

    status = {}

    runs = get_folder_list(root)
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

if __name__ == "__main__":

    root = "root"
    status = get_file_status(root)
    print(status)

