from pathlib import Path


class StorageHandler:
    def __init__(self, root_dir):
        self.root_dir: Path = Path(root_dir)
        self.dirs: dict[str, dict[int, set[int]]] = self._parse_directory_structure()

    def _parse_directory_structure(self) -> dict[str, dict[int, set[int]]]:
        """walk through root get structure of storage"""
        dirs: dict[str, dict[int, set[int]]] = {}
        for run_dir in self.root_dir.glob("run=[0-9][0-9][0-9]"):
            run_id: int = int(run_dir.name.split("=")[1])
            dirs[run_id] = {}

            for scan_dir in run_dir.glob("scan=[0-9][0-9][0-9]"):
                scan_id: int = int(scan_dir.name.split("=")[1])
                dirs[run_id][scan_id] = set()

                for p_dir in scan_dir.glob("p[0-9][0-9][0-9][0-9].h5"):
                    p_id: int = int(p_dir.stem[1:])
                    dirs[run_id][scan_id].add(p_id)

        return dirs


if __name__ == "__main__":
    from src.config.config import load_config

    config = load_config()
    load_dir: str = config.path.load_dir
    print(f"{load_dir = }")
    storage = StorageHandler(load_dir)
    print(storage.dirs)
