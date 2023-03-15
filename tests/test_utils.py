import os

from nlinec.utils import get_data_dir, get_models_dir, get_results_dir


def test_get_data_dir() -> None:
    data_dir = get_data_dir()

    assert os.path.exists(data_dir)


def test_get_models_dir() -> None:
    models_dir = get_models_dir()

    assert os.path.exists(models_dir)


def test_get_results_dir() -> None:
    results_dir = get_results_dir()

    assert os.path.exists(results_dir)
