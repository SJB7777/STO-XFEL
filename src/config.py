from cuptlib_config.palxfel import load_palxfel_config, save_palxfel_dict, ExperimentConfiguration

config_dir = "config\isaac\config.ini"

def load_config() -> ExperimentConfiguration:
    return load_palxfel_config(config_dir)

def save_config(config_dict: dict) -> None:
    save_palxfel_dict(config_dict, config_dir)