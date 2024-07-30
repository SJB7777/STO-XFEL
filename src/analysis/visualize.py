import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    from analysis.mean_data_processor import MeanDataProcessor
    from roi_rectangle import RoiRectangle
    # npz_file = "D:\\dev\\p_python\\xrd\\xfel_sample_data\\Npz_files\\run=0001_scan=0001.npz"
    npz_file = "D:\\dev\\p_python\\xrd\\xfel_sample_data\\Npz_files\\run=062\\scan=001\\run=062, scan=001.npz"
    mdp = MeanDataProcessor(npz_file)
    roi_rects = [RoiRectangle(0, 0, None, None), RoiRectangle(100, 200, 500, 600)]
    names = ["total", "center"]
    named_roi_rects = zip(names, roi_rects)
    data_df = mdp.analyze_by_rois(named_roi_rects)

    name = 'total'
    
    delays = data_df.index
    poff_intensities = data_df[name]["poff_com_x"]
    pon_intensities = data_df[name]["pon_com_x"]

    # 그래프 생성
    plt.figure(figsize=(10, 6))  # 그래프 크기 설정

    # 데이터 플로팅
    plt.plot(delays, poff_intensities, label='POFF Intensities', color='blue', linestyle='-', marker='o', markersize=4)
    plt.plot(delays, pon_intensities, label='PON Intensities', color='red', linestyle='--', marker='s', markersize=4)

    # 그래프 레이블 및 제목 추가
    plt.xlabel('Delays(ps)', fontsize=14, fontname='Arial')
    plt.ylabel('Intensities(a.u.)', fontsize=14, fontname='Arial')
    plt.title('Intensities over Delays', fontsize=16, fontname='Arial')

    # 범례 추가
    plt.legend(loc='upper left', fontsize=12)

    # 그리드 추가
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 틱 포맷 설정
    plt.xticks(fontsize=12, fontname='Arial')
    plt.yticks(fontsize=12, fontname='Arial')

    # 배경색 설정 (투명하게 설정)
    plt.gca().set_facecolor('none')

    # 그래프 표시
    plt.tight_layout()  # 레이아웃 조정
    plt.savefig('graph.png', dpi=300, bbox_inches='tight')  # 고해상도로 저장
    plt.show()