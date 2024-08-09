import tkinter as tk
from tkinter import Scale
from PIL import Image, ImageTk
import os

import numpy as np
from roi_rectangle import RoiRectangle
from analysis.mean_data_processor import MeanDataProcessor

# file = "D:\\dev\\p_python\\xrd\\xfel_sample_data\\Npz_files\\run=0001_scan=0001.npz"
file = "D:\\dev\\p_python\\xrd\\xfel_sample_data\\Npz_files\\run=062\\scan=001\\run=062, scan=001.npz"
mdp = MeanDataProcessor(file, -45)

images = np.log1p(mdp.pon_images)
images = images / images.max()
images = (images * 255).astype(np.uint8)  # 0-255 범위로 스케일링

# Tkinter GUI 설정
root = tk.Tk()
root.title("Image Viewer")

# 이미지 표시를 위한 Label
image_label = tk.Label(root)
image_label.pack()

# 이미지 인덱스 초기화
image_index = 0

# 이미지 로드 및 표시 함수
def show_image(index):
    global images
    if 0 <= index < len(images):
        image = images[index]
        image = Image.fromarray(image)  # NumPy 배열을 PIL 이미지로 변환
        tk_image = ImageTk.PhotoImage(image)
        image_label.config(image=tk_image)
        image_label.image = tk_image

# 슬라이더 값 변경 시 이미지 업데이트
def on_scale_change(value):
    global image_index
    image_index = int(value)
    show_image(image_index)

# 슬라이더 설정
scale = Scale(root, from_=0, to=len(images) - 1, orient=tk.HORIZONTAL, command=on_scale_change)
scale.pack()

# 초기 이미지 표시
show_image(image_index)

# Tkinter 메인 루프 실행
root.mainloop()