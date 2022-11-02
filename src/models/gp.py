"""The simplest possible Gaussian Process."""
from typing import NewType, Optional, Tuple

import numpy as np
import tensorflow as tf

ColumnVector = NewType("ColumnVector", np.ndarray)  # Shape (n, 1)
ColumnTensor = NewType("ColumnTensor", tf.Tensor)  # type: ignore  # (n, 1)


class GP:
    """
    A terrible implementation of a Gaussian Process with 0 mean function and
    the exponentiated quadratic kernel.

    f ~ GP(mu, K)

    m(x) = 0
    k(x1, x2) = s * e^(-|x1 - x2|^2/(2*l^2))

    Caveats::

    * noise-free observations,
    * single input dimension, single output dimension.
    """

    def __init__(self, lengthscale: float = 1.0, s: float = 0.1) -> None:
        """
        Define the learned parameters
        """
        # no mean function, but add variables here in the future

        # NOTE: the kernel function is screaming for an abstraction
        # two variables for the squared exponential
        self._l = tf.Variable(
            initial_value=lengthscale,
            trainable=True,
            dtype=tf.float64,
            name="lengthscale",
        )
        self._s = tf.Variable(
            initial_value=s, trainable=True, dtype=tf.float64, name="s"
        )
        self._trainable_variables = [self._l, self._s]

        self._xt: ColumnTensor = tf.constant(float("nan"), dtype=tf.float64)
        self._yt: ColumnTensor = tf.constant(float("nan"), dtype=tf.float64)

    def fit(
        self,
        x: ColumnVector,
        y: ColumnVector,
        steps: int = 1_000,
        learning_rate: float = 0.1,
        print_debug_messages: bool = False,
        debug_info_interval: Optional[int] = None,
    ) -> None:
        debug_info_interval_: int = (
            debug_info_interval
            if debug_info_interval
            else steps // 10
            if steps > 9
            else 1
        )

        # Convert input to tensors
        xt = tf.constant(x, dtype=tf.float64)
        yt = tf.constant(y, dtype=tf.float64)
        h = tf.constant(learning_rate, dtype=tf.float64)

        # Store the fitting inputs - needed for predictions
        self._xt = xt
        self._yt = yt

        # @tf.function
        def _step() -> None:
            with tf.GradientTape() as g:
                loss = self._loss(xt, yt)

            gradients = g.gradient(loss, self._trainable_variables)

            # Now apply gradients to the variables -
            # simplest possible optimiser
            for grad, variable in zip(gradients, self._trainable_variables):
                gradient_update = h * grad
                variable.assign_sub(gradient_update)

        for i in range(steps):
            intermediate_loss = self._loss(xt, yt)
            _step()
            if i % debug_info_interval_ == 0 and print_debug_messages:
                print(f"Step {i}")
                print(self.debug_message())
                print(f"Loss: {intermediate_loss.numpy()}")

    def _loss(self, x: ColumnTensor, y: ColumnTensor) -> tf.Tensor:
        """
        Maximise the likelihood estimate.

        If Y ~ N(m, K) with n-dimensional Y and
        if we denote all m, K params by w, then
        p_w(Y=y; x) =
        (2*pi)^{-n/2} * det(K(x, x; w))^{-1/2} *
        * exp(-(y-m(x; w))^T K(x, x; w)^{-1} (y-m(x; w))/2)

        We want to maximise over w, so the constant multiple doesn't matter.

        Similarly, argmax_w {p_w(X=x)} = argmax_w {g(p_w(X=x))}
        for any monotonic g.
        Since we're dealing with exp, let's pick log.

        Applying standard log rules shows us that we're maximising
        -log(det(K(x, x; w)))/2 - (y-m(x; w))^T K(x, x; w)^{-1} (y-m(x; w))/2

        Since our optimiser is minimising, we want to minimise
        log(det(K(x, x; w)))/2 + (y-m(x; w))^T K(x, x; w)^{-1} (y-m(x; w))/2
        with respect to parameters w.
        """
        kk = self.cov(x, x)  # the params w are implicit in k
        m = self.m(x)  # the params w are implicit in m
        det_kk = tf.linalg.det(kk)

        chol_kk = tf.linalg.cholesky(kk)  # TODO: implement
        # NOTE: the following is slow and bad.
        # For real applications you should simplify
        # the inner product using the Cholesky decomposition of K(x, x; w).
        inv_k = tf.linalg.inv(kk)
        inner_prod_term = tf.linalg.matmul(
            a=y - m,
            b=tf.linalg.matmul(inv_k, y - m),
            transpose_a=True,
        )
        min_det = 1e-6
        double_log_likelihood = tf.math.log(det_kk + min_det) + inner_prod_term
        return double_log_likelihood

    def debug_message(self) -> str:
        ret = ""
        for var in self._trainable_variables:
            ret += f"{var.name}: {var.numpy()}\n"
        return ret

    def _predict(self, x_new: ColumnTensor) -> ColumnTensor:
        """
        Predict the mean of the distribution at x.
        """
        m, cov = self._conditional_distribution_at(x_new)
        return m

    def _conditional_distribution_at(
        self, x_new: ColumnTensor
    ) -> Tuple[ColumnTensor, ColumnTensor]:
        """
        Return the conditional distribution p(y, x| xt, yt).
        It is a MVN distribution, so characterised by the mean and covariance,
        which we return here.

        The model assumes y, yt are sampled from
        a multivariate-normal distribution.
        The conditioning formulas for MVN are well known
        (and are derived by completing
        the square in joint distribution carefully).

        Let's write it down explicitly.
        Define k_ as the row vector (K(x, xt_1), K(x, xt_2), ..., K(x, xt_n))
        of shape (1, n).
        Then Y(x) ~ N(mp, Kp) where
        mp := m(x) + k_ * K(xt, xt)^{-1} * (yt - m(x)), and
        Kp := K(x, x) - k_ * K(xt, xt)^{-1} * (k_^T).
        """
        xt, yt = self._xt, self._yt
        k_ = self.cov(x_new, xt)
        kk = self.cov(xt, xt)
        kk_new = self.cov(x_new, x_new)

        # The matrix inversion is numerically unstable,
        # so let's work with Cholesky decomp instead
        chol_kk = tf.linalg.cholesky(kk)
        inv_chol_kk = tf.linalg.inv(chol_kk)

        k_inv_chol_kkt = tf.linalg.matmul(a=k_, b=inv_chol_kk, transpose_b=True)
        inv_chol_kk_y_m = tf.linalg.matmul(inv_chol_kk, yt - self.m(xt))

        m_update = tf.linalg.matmul(k_inv_chol_kkt, inv_chol_kk_y_m)
        mp = self.m(x_new) + m_update
        kp = kk_new - tf.linalg.matmul(a=k_inv_chol_kkt, b=k_inv_chol_kkt, transpose_b=True)
        return mp, kp

    def conditional_distribution_at(
        self, x_new: ColumnVector
    ) -> Tuple[ColumnVector, ColumnVector]:
        """
        Return the conditional distribution p(y(x_new)| y_seen(x_sampled)).
        It is a MVN distribution, so characterised by the mean and covariance,
        which we return here.
        """
        x_new_t = tf.constant(x_new, dtype=tf.float64)
        m, cov = self._conditional_distribution_at(x_new_t)
        return m.numpy(), cov.numpy()

    def predict(self, x_new: ColumnVector) -> np.ndarray:
        """Return the mean of the distribution at x_new"""
        x_new_t = tf.constant(x_new, dtype=tf.float64)
        m = self._predict(x_new_t)
        return m.numpy()

    def m(self, x: ColumnTensor) -> ColumnTensor:
        """The mean function. Current implementation: 0."""
        return tf.zeros_like(x)

    def cov(self, x1: ColumnTensor, x2: ColumnTensor) -> tf.Tensor:
        """
        The covariance function.
        Current implementation: squared exponential.
        cov(x1, x2)_{i, j} = s * e^(-|x1_i - x2_j|^2/(2*l^2))

        Expects inputs of shape (n, 1) and (m, 1).
        Returns a covariance matrix of shape (n, m).
        """
        assert x2.shape.rank == 2
        m = x2.shape.as_list()[0]

        def positive_bij(x_: tf.Tensor) -> tf.Tensor:
            return tf.math.softplus(x_)

        x2_row = tf.reshape(x2, (1, m))
        kronecker_diff = x1 - x2_row  # Shape (n, m)
        distances_squared = tf.math.square(kronecker_diff)
        sd = positive_bij(self._s)
        lengthscale_squared = positive_bij(self._l) ** 2
        cov = sd * tf.math.exp(-distances_squared / (2 * lengthscale_squared))
        return cov


if __name__ == "__main__":
    from src.data.load import get_data

    tf.config.set_visible_devices(
        [], "GPU"
    )  # Ignore GPU - only needed on crappy laptops like mine :(

    df = get_data("solar")
    df.reset_index(inplace=True)
    gp = GP()
    x, y = df["x"].values.reshape(-1, 1), df["y"].values.reshape(-1, 1)

    # gp.fit(x=x, y=y, learning_rate=1e-9, steps=1)
    # preds = gp.predict(x)
    # print(f"Learned params (normalised): {gp.debug_message()}")

    gp_partial = GP(s=10000)
    x_p, y_p = x[::2], y[::2]
    gp_partial.fit(x=x_p, y=y_p, steps=0)
    preds_p = gp_partial.predict(x)
