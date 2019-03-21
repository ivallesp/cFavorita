import json



def get_general_config():
    """
    Loads the config in the settings.json file
    :return: a dictionary with all the configuration parameters
    """
    from src.common_paths import get_config_filepath
    filepath = get_config_filepath()
    config = json.load(open(filepath))
    return config
