from cuptlib_config.palxfel import save_palxfel_dict
from cuptlib_config.palxfel.enums import Hertz, Hutch, Detector, Xray

config_dict = {
    "path":{
        # Mother Directory of run files. 
        "load_dir": "D:/dev/p_python/xrd/xfel_sample_data",
        "save_dir": "D:/dev/p_python/xrd/xfel_sample_data",
        "image_dir": "Image",  # relative path based on save_dir
        "param_dir": "DataParameter",  # relative path based on save_dir
        "mat_dir": "Mat_files",
        "npz_dir": "Npz_files_qbpm",  # npz_files_qbpm
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