"""A silly TF implementation of the linear model that conforms to the fit-predict interface."""

import numpy as np
import tensorflow as tf


class LinearRegression:
    """
    A terrible implementation of linear regression.
    """

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, steps: int = 100) -> None:
        xt = tf.constant(x, dtype=tf.float64)
        yt = tf.constant(y, dtype=tf.float64)

        theta = tf.Variable(initial_value=0., trainable=True, dtype=tf.float64, name="theta")
        b = tf.Variable(initial_value=0., trainable=True, dtype=tf.float64, name="b")

        optimizer = tf.optimizers.SGD()

        @tf.function
        def _step(x_: tf.Tensor, y_: tf.Tensor, theta_: tf.Tensor, b_: tf.Tensor) -> None:
            with tf.GradientTape() as g:
                y_pred = theta_ * x_ + b_
                loss = tf.keras.losses.MSE(y_true=y_, y_pred=y_pred)

                gradients = g.gradient(loss, [theta_, b_])

                # Update theta
                optimizer.apply_gradients(zip(gradients, [theta_, b_]))

        for step in range(1, steps + 1):
            _step(xt, yt, theta, b)

            pred = theta * b
            intermediate_loss = tf.keras.losses.MSE(pred, yt)
            print(f"step: {step}, loss: {intermediate_loss}, theta: {theta.numpy()}, b: {b.numpy()}")

    def predict(self, x, y) -> None:
        pass


if __name__ == '__main__':
    from src.data.load import get_data

    tf.config.set_visible_devices([], 'GPU')  # Ignore GPU - only needed on crappy laptops like mine :(

    df = get_data("solar")
    df.reset_index(inplace=True)
    linreg = LinearRegression()
    linreg.fit(x=df["x"], y=df["y"])
