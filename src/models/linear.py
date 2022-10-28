"""A silly TF implementation of the linear model that conforms to the fit-predict interface."""

import numpy as np
import tensorflow as tf


class LinearRegression:
    """
    A terrible implementation of linear regression.
    """

    @staticmethod
    def fit(
        x: np.ndarray, y: np.ndarray, steps: int = 100, learning_rate: float = 0.01
    ) -> None:
        xt = tf.constant(x, dtype=tf.float64)
        yt = tf.constant(y, dtype=tf.float64)

        theta = tf.Variable(
            initial_value=(y.max() - y.min()) / (x.max() - x.min()),
            trainable=True,
            dtype=tf.float64,
            name="theta",
        )
        b = tf.Variable(
            initial_value=y.mean(), trainable=True, dtype=tf.float64, name="b"
        )

        h_theta = tf.constant(learning_rate * theta.numpy(), dtype=tf.float64)
        h_b = tf.constant(learning_rate * b.numpy(), dtype=tf.float64)

        # @tf.function
        def _step(theta_: tf.Tensor, b_: tf.Tensor) -> None:
            with tf.GradientTape() as g:
                y_pred = theta_ * xt + b_
                loss = tf.keras.losses.MSE(y_true=yt, y_pred=y_pred)

            gradients = g.gradient(loss, [theta_, b_])

            print(
                f"Before update. Loss: {loss.numpy()}, theta: {theta.numpy()}, "
                f"b: {b.numpy()}, gradients: {[grad.numpy() for grad in gradients]}"
            )

            # Now apply gradients to the variables
            dl_dtheta, dl_db = gradients
            theta_.assign_sub(h_theta * dl_dtheta)
            b_.assign_sub(h_b * dl_db)

            print(
                f"After update. Loss: {loss.numpy()}, theta: {theta.numpy()}, "
                f"b: {b.numpy()}, gradients: {[grad.numpy() for grad in gradients]}"
            )

        for step in range(steps + 1):
            pred = theta * x + b
            intermediate_loss = tf.keras.losses.MSE(pred, yt)
            _step(theta, b)

    def predict(self, x, y) -> None:
        pass


if __name__ == "__main__":
    from src.data.load import get_data

    tf.config.set_visible_devices(
        [], "GPU"
    )  # Ignore GPU - only needed on crappy laptops like mine :(

    df = get_data("solar")
    df.reset_index(inplace=True)
    linreg = LinearRegression()
    linreg.fit(x=df["x"], y=df["y"], learning_rate=0.00001)
