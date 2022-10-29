"""
A trivial loader for dummy datasets.

Source:
https://github.com/jamesrobertlloyd/gp-structure-search/tree
/master/data/1d_data
"""
from pathlib import Path

import pandas as pd
from scipy.io import loadmat

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = REPO_ROOT / "data"
REGISTERED_DATASETS = {
    "airline": "01-airline.mat",
    "solar": "02-solar.mat",
    "mauna": "03-mauna2003.mat",
    "methane": "04-methane.mat",
}


def get_data(identifier: str) -> pd.DataFrame:
    path = DATA_PATH / REGISTERED_DATASETS[identifier]
    data = loadmat(str(path))
    df = pd.DataFrame({"x": data["X"].flatten(), "y": data["y"].flatten()})
    df.set_index("x", inplace=True)
    return df


if __name__ == "__main__":
    print(get_data("mauna").describe())
