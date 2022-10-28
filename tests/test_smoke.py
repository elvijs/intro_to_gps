"""Simple smoke tests to alert us to incomplete setup."""

from src.data.load import get_data, REGISTERED_DATASETS


def test_datasets() -> None:
    for id_ in REGISTERED_DATASETS.keys():
        get_data(id_)
