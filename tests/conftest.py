import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def run_on_cpu() -> None:
    tf.config.set_visible_devices([], "GPU")
