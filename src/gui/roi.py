import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Button

# 전역 변수 선언
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
rect = None
ax = None

def on_mouse_press(event):
    global ix, iy, fx, fy, drawing, rect, ax

    if event.inaxes is not None:
        if event.button == 1:  # 왼쪽 마우스 버튼
            drawing = True
            ix, iy = int(event.xdata), int(event.ydata)
            fx, fy = ix, iy
            if rect is not None:
                rect.remove()
            rect = patches.Rectangle((ix, iy), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.draw()

def on_mouse_release(event):
    global fx, fy, drawing, rect

    if event.inaxes is not None and drawing:
        drawing = False
        fx, fy = int(event.xdata), int(event.ydata)
        if rect is not None:
            rect.set_width(fx - ix)
            rect.set_height(fy - iy)
            plt.draw()

def on_mouse_move(event):
    global fx, fy, drawing, rect

    if event.inaxes is not None and drawing:
        fx, fy = int(event.xdata), int(event.ydata)
        if rect is not None:
            rect.set_width(fx - ix)
            rect.set_height(fy - iy)
            plt.draw()

def select_roi(image):
    global ix, iy, fx, fy, rect, ax

    # 이미지 복사
    img = image.copy()

    # 윈도우 생성
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    # 마우스 이벤트 콜백 함수 등록
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # ROI 선택 완료를 위한 플래그
    roi_selected = False

    plt.show()

    if ix == -1 or iy == -1 or fx == -1 or fy == -1:
        return None
    else:
        # 좌표 정렬
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        return (x1, y1, x2, y2)

if __name__ == "__main__":
    from core.loading_strategy import HDF5FileLoader
    file: str = "Y:\\240608_FXS\\raw_data\\h5\\type=raw\\run=176\\scan=001\\p0041.h5"
    hfl = HDF5FileLoader(file)
    image = np.log1p(hfl.images.sum(axis=0))
    image = image.astype(np.float32)
    if image is None:
        print("이미지를 불러올 수 없습니다.")
    else:
        roi_coords = select_roi(image)
        if roi_coords is not None:
            print(f"선택된 ROI 좌표: {roi_coords}")
        else:
            print("ROI가 선택되지 않았습니다.")