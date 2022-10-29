"""Simple smoke tests for linear regression."""
import pandas as pd
import pytest

from src.data.load import REGISTERED_DATASETS, get_data
from src.models.linear import LinearRegression


@pytest.fixture(name="data", params=REGISTERED_DATASETS.keys())
def _data(request) -> pd.DataFrame:
    dataset = request.param
    df = get_data(dataset)
    df.reset_index(inplace=True)
    return df


def test_linear_regression__is_not_smoking(data) -> None:
    model = LinearRegression()
    model.fit(data["x"], data["y"])
    model.predict(data["x"])
