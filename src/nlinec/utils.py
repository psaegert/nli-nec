import os


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
