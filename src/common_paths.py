import json
import os

from src.general_utilities import get_general_config


def _norm_path(path):
    """
    Decorator function intended for using it to normalize a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """

    def normalize_path(*args, **kwargs):
        return os.path.normpath(path(*args, **kwargs))

    return normalize_path


def _assure_path_exists(path):
    """
    Decorator function intended for checking the existence of a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """

    def assure_exists(*args, **kwargs):
        p = path(*args, **kwargs)
        assert os.path.exists(p), "the following path does not exist: '{}'".format(p)
        return p

    return assure_exists


def _is_output_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an output path retrieval
    function
    """

    @_norm_path
    @_assure_path_exists
    def check_existence_or_create_it(*args, **kwargs):
        if not os.path.exists(path(*args, **kwargs)):
            "Path does not exist... creating it: {}".format(path(*args, **kwargs))
            os.makedirs(path(*args, **kwargs))
        return path(*args, **kwargs)

    return check_existence_or_create_it


def _is_input_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an input path retrieval
    function
    """

    @_norm_path
    @_assure_path_exists
    def check_existence(*args, **kwargs):
        return path(*args, **kwargs)

    return check_existence


@_is_input_path
def get_config_filepath():
    """
    Generates the path where the general config json file is located
    :return: settings.json file path (str)
    """
    filepath = "settings.json"
    return filepath


@_is_input_path
def get_data_path():
    """
    Generates the path where the raw data is located
    :return: data folder path (str)
    """
    config = get_general_config()
    return config["paths"]["data"]


@_is_output_path
def get_tensorboard_path():
    """
    Generates the path where the tensorboard logs will be stored
    :return: data folder path (str)
    """
    config = get_general_config()
    return config["paths"]["tensorboard"]


@_is_input_path
def get_log_config_filepath():
    return "logging_config.ini"


@_is_output_path
def get_model_path(alias):
    """
    Generates the path where the tensorboard logs will be stored
    :return: data folder path (str)
    """
    return os.path.join("models", alias)
