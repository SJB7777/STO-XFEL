from src.gui.roi import select_roi_by_run_scan

if __name__ == "__main__":
    roi_rect = select_roi_by_run_scan(144, 1, 0)
    print(roi_rect)
