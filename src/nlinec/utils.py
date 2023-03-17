import os

import numpy as np


def get_data_dir() -> str:
    """
    Get the path to the data directory.

    Returns
    -------
    data_dir : str
        The path to the data directory.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    # Create the data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    return data_dir


def get_models_dir() -> str:
    """
    Get the path to the models directory.

    Returns
    -------
    models_dir : str
        The path to the models directory.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

    # Create the data directory if it does not exist
    os.makedirs(models_dir, exist_ok=True)

    return models_dir


def get_results_dir() -> str:
    """
    Get the path to the results directory.

    Returns
    -------
    results_dir : str
        The path to the results directory.
    """
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

    # Create the data directory if it does not exist
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def get_labels() -> dict:
    """
    Get the labels for the NLI tasks.

    Returns
    -------
    labels : dict
        The labels for the NLI tasks.
    """
    labels_path = os.path.join(get_data_dir(), 'labels.json')

    if os.path.exists(labels_path):
        import json
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        labels = model.config.label2id

        with open(labels_path, 'w') as f:
            import json
            json.dump(labels, f)

    return labels


def color_palette(key: str | None = None) -> str | dict:
    """
    Get the color palette.

    Returns
    -------
    colors : dict
        The color palette.
    """
    colors = {
        "dark": "#232526",
        "medium": "#394a59",
        "light": "#aed8f2",
        "bright": "#f2f2f2",
        "accent": "#f29727"
    }

    if key is None:
        return colors

    return colors[key]


def gpartconv1d(data: np.ndarray, sigma: float, window_size: int = None) -> np.ndarray:
    '''
    Convolve data with a gaussian kernel and the edges with a truncated gaussian kernel.

    Parameters
    ----------
    data : np.ndarray
        Data to be convolved.
    sigma : float
        Standard deviation of the gaussian kernel.
    window_size : int, optional
        Size of the window. The default is None.

    Returns
    -------
    np.ndarray
        Convolved data.
    '''
    if window_size is None:
        window_size = min(int(sigma * 3.5), (len(data) - 1) // 2)

    # Create the gaussian kernel
    kernel = np.exp(-np.arange(-window_size, window_size + 1)**2 / (2 * sigma**2))

    # Convolve the middle section of the data with the kernel, i.e. where the data and the kernel overlap completely
    data_convolved_middle = np.convolve(data, kernel / kernel.sum(), mode='valid')

    # Convolve the edges of the data with the kernel, i.e. where the data and the kernel overlap partially
    data_convolved_left = np.empty(2 * window_size - 1)
    data_convolved_right = np.empty(2 * window_size - 1)
    for i in range(1, 2 * window_size):
        data_convolved_left[i - 1] = data[:i] @ kernel[-i:] / kernel[-i:].sum()
        data_convolved_right[i - 1] = data[-2 * window_size + i:] @ kernel[:2 * window_size - i] / kernel[:2 * window_size - i].sum()

    # Convolve the data with the kernel
    data_convolved = np.concatenate((data_convolved_left[window_size - 1:], data_convolved_middle, data_convolved_right[:window_size]))

    return data_convolved
