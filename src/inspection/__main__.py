import os

import click

from src.config.config import load_config
from src.inspection.hdf5_validator import check_scan
from src.utils.file_util import get_run_scan_directory


@click.group()
def cli():
    """Command line tool for managing hierarchical file structures."""
    pass


@cli.command()
@click.argument('run_n', type=int)
@click.option('--size', is_flag=True, help='Show detailed size information')
def run(run_n: int, size: bool) -> None:
    """List files of the run directory"""
    click.echo(f'run={run_n:03}')
    config = load_config()
    load_dir = config.path.load_dir
    run_dir = get_run_scan_directory(load_dir, run_n)
    if not os.path.exists(run_dir):
        click.echo(f'No such file or directory: {run_dir}')
        return None
    scan_names: list[str] = os.listdir(run_dir)
    scan_names.sort()
    for scan_name in scan_names:
        scan_dir = os.path.join(run_dir, scan_name)
        file_names: list[str] = os.listdir(scan_dir)
        file_names.sort()
        for file_name in file_names:
            line = [file_name]

            if size:
                file = os.path.join(scan_dir, file_name)
                size = os.path.getsize(file)
                line.append(f'size {size} (bytes)')

            click.echo(' '.join(line))


if __name__ == '__main__':
    cli()
