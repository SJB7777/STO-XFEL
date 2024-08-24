import os


def check_scan(scan_dir: str) -> dict[str, int]:
    file_names: list[str] = os.listdir(scan_dir)
    file_names.sort(key=lambda name: int(name[1:-3]))
    sizes: dict[str, int] = {}

    for file_name in file_names:
        file = os.path.join(scan_dir, file_name)
        size = os.path.getsize(file)
        sizes[file_name] = size
    
    return sizes


if __name__ == '__main__':
    from src.config.config import load_config
    from src.utils.file_util import get_run_scan_directory
    config = load_config()
    load_dir: str = config.path.load_dir

    run_n: int = int(input("run number: "))
    scan_dir: str = get_run_scan_directory(load_dir, run_n, 1)

    sizes = check_scan(scan_dir)

    for file_name, size in sizes.items():
        print(f'{file_name} {size} bytes')
