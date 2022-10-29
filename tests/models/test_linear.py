"""Simple smoke tests for linear regression."""
import numpy as np
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
    model.fit(data["x"].values, data["y"].values)
    preds = model.predict(data["x"].values)
    mean_error = np.mean(np.abs(preds - data["y"]))
    assert mean_error < 40  # high mean error, this data isn't linear


@pytest.mark.parametrize(
    "x, y, expected_y",
    [
        (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
        ),  # corresponds to theta=1, b=0
        (np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([1, 2, 3])),  # theta=1, b=1
        (
            np.array([0, 1, 2]),
            np.array([1 - 0.1, 2 + 0.2, 3 - 0.1]),
            np.array([1, 2, 3]),
        ),  # theta~=1, b~=1, but with noise
    ],
)
def test_linear_regression__on_trivial_data(
    x: np.ndarray, y: np.ndarray, expected_y: np.ndarray
) -> None:
    linreg = LinearRegression()
    linreg.fit(x=x, y=y, steps=1_000)

    preds = linreg.predict(x)

    np.testing.assert_array_almost_equal(preds, expected_y)
