import os


def get_data_dir(*args: str) -> str:
    """
    Get the path to the data directory.

    Parameters
    ----------
    *args : str
        The path to file or directory in the data directory.

    Returns
    -------
    data_dir : str
        The path to the data directory.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', *args)

    # Create the data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    return data_dir


def get_models_dir(*args: str) -> str:
    """
    Get the path to the models directory.

    Parameters
    ----------
    *args : str
        The path to file or directory in the models directory.

    Returns
    -------
    models_dir : str
        The path to the models directory.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', *args)

    # Create the data directory if it does not exist
    os.makedirs(models_dir, exist_ok=True)

    return models_dir


def get_results_dir(*args: str) -> str:
    """
    Get the path to the results directory.

    Parameters
    ----------
    *args : str
        The path to file or directory in the results directory.

    Returns
    -------
    results_dir : str
        The path to the results directory.
    """
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', *args)

    # Create the data directory if it does not exist
    os.makedirs(results_dir, exist_ok=True)

    return results_dir
