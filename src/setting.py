from cuptlib_config.palxfel import save_palxfel_dict
from cuptlib_config.palxfel.enums import Hertz, Hutch, Detector, Xray

config_dict = {
    "path":{
        # Mother Directory of run files. 
        "load_dir": "Y:\\240608_FXS\\raw_data\\h5\\type=raw",
        "save_dir": "Y:\\240608_FXS\\raw_data\\h5\\type=raw",
        "image_dir": "Image",  # relative path based on save_dir
        "param_dir": "DataParameter",  # relative path based on save_dir
        "mat_dir": "Mat_files2",
        "npz_dir": "Npz_files",  # npz_files_qbpm
        "tif_dir": "Tif_files"
    },
    "param":{
        # Hutch 
        "hutch": Hutch.EH1,
        # Detector
        "detector": Detector.JUNGFRAU2,
        # Xray used in experiment. 
        "xray": Xray.HARD,
        # Rate of laser.
        "pump_setting": Hertz.FIFTEEN,
        # Index of roi coordinate inside h5 file.
        "x1": 0, "x2": 1, "y1": 2, "y2": 3,
        # Metric of SDD and DPS is meters.
        "sdd": 1.3,
        "dps": 7.5e-5, # Detector Pixel Size
        "beam_energy": 9.7,
    }
}

def save():
    save_palxfel_dict(config_dict, "config.ini")

if __name__ == "__main__":
    save()