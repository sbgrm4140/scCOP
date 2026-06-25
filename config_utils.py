import yaml
import os


def load_config(config_path="config.yaml"):
    """
    Load configuration parameters from a YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_param(config, param_path, default=None):
    """
    Retrieve a nested parameter from the configuration dictionary.

    Args:
        config (dict): Configuration dictionary.
        param_path (str): Parameter path, for example "model.hidden_dims".
        default: Default value returned if the parameter does not exist.

    Returns:
        The parameter value or the default value.
    """
    keys = param_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default