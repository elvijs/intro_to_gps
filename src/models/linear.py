"""A silly TF implementation of the linear model that conforms to the fit-predict interface."""
from typing import Optional

import numpy as np
import tensorflow as tf


class LinearRegression:
    """
    A terrible implementation of linear regression.

    y = theta * x + b  (1D theta!!)
    """

    def __init__(self):
        self._theta: Optional[tf.Variable] = None
        self._b: Optional[tf.Variable] = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        steps: int = 100,
        learning_rate: float = 1e-6,
        debug_info_interval: Optional[int] = None,
    ) -> None:
        debug_info_interval = debug_info_interval or steps / 10

        # convert input to tensors
        xt = tf.constant(x, dtype=tf.float64)
        yt = tf.constant(y, dtype=tf.float64)
        h = tf.constant(learning_rate, dtype=tf.float64)

        # sensible initial values
        b = tf.Variable(
            initial_value=y.mean(),
            trainable=True, dtype=tf.float64, name="b"
        )
        self._b = b
        theta = tf.Variable(
            initial_value=(y.max() - y.min()) / (x.max() - x.min()),
            trainable=True,
            dtype=tf.float64,
            name="theta",
        )
        self._theta = theta

        @tf.function
        def _step(theta_: tf.Tensor, b_: tf.Tensor) -> None:
            with tf.GradientTape() as g:
                y_pred = theta_ * xt + b_
                loss = tf.sqrt(tf.keras.losses.MSE(y_true=yt, y_pred=y_pred))

            gradients = g.gradient(loss, [theta_, b_])

            # Now apply gradients to the variables
            dl_dtheta, dl_db = gradients
            theta_.assign_sub(h * dl_dtheta)
            b_.assign_sub(h * dl_db)

        for i in range(steps + 1):
            pred = theta * x + b
            intermediate_loss = tf.sqrt(tf.keras.losses.MSE(pred, yt))
            _step(theta, b)
            if i % debug_info_interval == 0:
                print(
                    f"Step {i}. "
                    f"Loss: {intermediate_loss.numpy()}, "
                    f"theta: {theta.numpy()}, "
                    f"b: {b.numpy()}"
                )

    def predict(self, x: np.ndarray) -> None:
        return self._theta.numpy() * x + self._b.numpy()


if __name__ == "__main__":
    from src.data.load import get_data

    tf.config.set_visible_devices(
        [], "GPU"
    )  # Ignore GPU - only needed on crappy laptops like mine :(

    df = get_data("solar")
    df.reset_index(inplace=True)
    linreg = LinearRegression()
    linreg.fit(x=df["x"], y=df["y"], steps=1_000, learning_rate=1e-8)

    df["preds"] = linreg.predict(df["x"])
    print(df)

