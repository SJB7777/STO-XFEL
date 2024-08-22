from src.gui.roi import select_roi_by_run_scan

if __name__ == "__main__":
    roi_rect = select_roi_by_run_scan(43, 1, 1)
    print(roi_rect)
