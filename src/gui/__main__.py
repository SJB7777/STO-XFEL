import os

import numpy as np
from roi_rectangle import RoiRectangle
import click

from src.gui.roi import RoiSelector
from src.config.config import load_config
from src.analyzer.loader import NpzLoader


def select_roi(run_n: int) -> tuple[int, int, int, int]:
    """Select roi"""
    config = load_config()
    npz_file = os.path.join(config.path.npz_dir, f"run={run_n:0>4}_scan=0001.npz")
    image = NpzLoader(npz_file).data['poff'].sum(0)
    roi = RoiSelector().select_roi(np.log1p(image))
    return roi


@click.command()
@click.argument('run_n', type=str)
@click.option('--roi', type=bool)
def gui_cli(run_n: int, is_roi: bool) -> None:
    """
    Command-line interface function to handle commands.

    Args:
        arg1 (str): The first argument.
        run_n (int): The run number.
    """
    if is_roi == 'roi':
        roi = select_roi(run_n)
        roi_rect = RoiRectangle.from_tuple(roi)
        click.echo(str(roi_rect))
    else:
        raise click.UsageError('Invalid command. Please use "roi" as the first argument.')

if __name__ == '__main__':
    gui_cli()
