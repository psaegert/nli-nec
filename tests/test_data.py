import os

from nlinec.data.load import get_data_dir


def test_get_data_dir() -> None:
    data_dir = get_data_dir()

    assert os.path.exists(data_dir)
