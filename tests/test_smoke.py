"""Simple smoke tests to alert us to incomplete setup."""

from src.data.load import REGISTERED_DATASETS, get_data


def test_datasets() -> None:
    for id_ in REGISTERED_DATASETS.keys():
        get_data(id_)
