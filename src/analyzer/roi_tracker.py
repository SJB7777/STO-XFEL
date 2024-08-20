from typing import Mapping

import numpy as np
import numpy.typing as npt
from roi_rectangle import RoiRectangle
from matplotlib import patches

from src.gui.roi import RoiSelector
from src.config.config import load_config, ExpConfig


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt

    run = 154
    scan = 1

    config: ExpConfig = load_config()
    npz_dir: str = config.path.npz_dir
    npz_file = os.path.join(npz_dir, f"run={run:0>4}_scan={scan:0>4}.npz")
    data: Mapping[str, npt.NDArray] = np.load(npz_file)
    delays: npt.NDArray = data["delay"]
    images: npt.NDArray = data["pon"]

    init_roi = RoiSelector().select_roi(np.log1p(images[0]))
    init_roi_rect: RoiRectangle = RoiRectangle.from_tuple(init_roi)
    roi_rects: list[RoiRectangle] = [init_roi_rect]

    for image in images:
        roi_image = roi_rects[-1].slice(image)
        com = np.unravel_index(np.argmax(roi_image), roi_image.shape)[::-1]
        new_center = (
            round(roi_rects[-1].x1 + com[0]),
            round(roi_rects[-1].y1 + com[1])
        )
        new_roi_rect = init_roi_rect.move_to_center(new_center)
        roi_rects.append(new_roi_rect)

    xs = []
    ys = []
    for roi_rect, image in zip(roi_rects[1:], images):
        roi_rect.slice(image)
        com = np.unravel_index(np.argmax(roi_image), roi_image.shape)[::-1]
        x = roi_rect.x1 + com[0]
        y = roi_rect.y1 + com[1]

        xs.append(x)
        ys.append(y)

    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(delays, xs)
    # axs[1].plot(delays, ys)

    # plt.show()

    for idx, image in enumerate(np.log1p(images)):

        # 이미지와 ROI를 그릴 그림 생성
        fig, ax = plt.subplots()
        ax.imshow(image)

        # 해당 이미지의 ROI 정보 가져오기
        roi_rect = roi_rects[idx]
        rect = patches.Rectangle((roi_rect.x1, roi_rect.y1), roi_rect.width, roi_rect.height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # 이미지와 ROI를 파일로 저장
        output_path = os.path.join("Y:/240608_FXS/raw_data/h5/type=raw/Image/moving_roi", f"image_{idx}.png")
        plt.savefig(output_path)
        plt.close(fig)
