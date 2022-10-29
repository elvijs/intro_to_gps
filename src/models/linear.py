"""A silly TF implementation of the linear model that conforms to the fit-predict interface."""
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class LinearRegression:
    """
    A terrible implementation of linear regression.

    y = theta * x + b  (scalar theta!!)
    """

    def __init__(self, theta: float = 1., b: float = 0.):
        self._theta = tf.Variable(
            initial_value=theta,
            trainable=True,
            dtype=tf.float64,
            name="theta",
        )
        self._b = tf.Variable(
            initial_value=b, trainable=True, dtype=tf.float64, name="b"
        )
        self._x_norm = StandardScaler()
        self._y_norm = StandardScaler()

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        steps: int = 1_000,
        learning_rate: float = 0.1,
        print_debug_messages: bool = False,
        debug_info_interval: Optional[int] = None,
    ) -> None:
        debug_info_interval = debug_info_interval or steps / 10

        # Convert input to tensors and normalise
        # Note the annoying reshapes and flattens as scikit-learn expects column vectors
        self._x_norm.fit(x.reshape(-1, 1))
        self._y_norm.fit(y.reshape(-1, 1))
        xt = tf.constant(self._x_norm.transform(x.reshape(-1, 1)).flatten(), dtype=tf.float64)
        yt = tf.constant(self._y_norm.transform(y.reshape(-1, 1)).flatten(), dtype=tf.float64)
        h = tf.constant(learning_rate, dtype=tf.float64)

        @tf.function
        def _step() -> None:
            with tf.GradientTape() as g:
                loss = self._loss(y_true=yt, y_pred=self._predict_normalised(xt))

            vars = [self._theta, self._b]
            gradients = g.gradient(loss, vars)

            # Now apply gradients to the variables
            dl_dtheta, dl_db = gradients
            self._theta.assign_sub(h * dl_dtheta)
            self._b.assign_sub(h * dl_db)

        for i in range(steps + 1):
            intermediate_loss = self._loss(y_pred=self._predict_normalised(xt), y_true=yt)
            _step()
            if i % debug_info_interval == 0 and print_debug_messages:
                print(f"Step {i}")
                print(self.debug_message())
                print(f"Loss: {intermediate_loss.numpy()}")

    def _predict_normalised(self, x: tf.Tensor) -> tf.Tensor:
        prediction = self._theta * x + self._b
        return prediction

    @staticmethod
    def _loss(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.MSE(y_pred, y_true)

    def debug_message(self) -> str:
        return f"theta: {self._theta.numpy()}, " f"b: {self._b.numpy()}"

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_n = self._x_norm.transform(x.reshape(-1, 1))
        y_n = self._theta.numpy() * x_n + self._b.numpy()
        return self._y_norm.inverse_transform(y_n).flatten()


if __name__ == "__main__":
    from src.data.load import get_data

    tf.config.set_visible_devices(
        [], "GPU"
    )  # Ignore GPU - only needed on crappy laptops like mine :(

    df = get_data("mauna")
    df.reset_index(inplace=True)
    linreg = LinearRegression(theta=10, b=10)
    linreg.fit(x=df["x"].values, y=df["y"].values)
    preds = linreg.predict(df["x"].values)
    print(f"Learned params (normalised), theta: {linreg._theta.numpy()}, b: {linreg._b.numpy()}")
