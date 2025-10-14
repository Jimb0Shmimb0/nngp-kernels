import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils.validation import check_array


class FiniteCosineActivationKernel(Kernel):
    """
    Finite-dimensional approximation for kernel definition: exp(-0.5(||x1||^2 + ||x2||^2)) * cosh(x1^T x2).

    Approximate using num_random_features samples

    note dim_x is the number of features
    """

    def __init__(self, X, num_random_features=100, random_state=None):
        self.num_random_features = num_random_features
        self.random_state = random_state

        self.W = np.random.RandomState(self.random_state).normal(
            loc=0.0,
            scale=1.0,
            size=(self.num_random_features, X.shape[1])
        )  # w_i \sim \mathcal{N}(0, I)

        # Note: If we want weights to be initialised during instantiation of this kernel, input X has to be passed in
        # W needs to match the shape of X.

        self.X = X

    def __call__(self, X, Y=None, eval_gradient=False):
        # Check X and Y are arrays
        X = check_array(X)
        if Y is None:
            Y = X
        else:
            Y = check_array(Y)

        Phi_X = np.cos(X @ self.W.T)
        Phi_Y = np.cos(Y @ self.W.T)

        # Compute inner product in the feature space, then divide by 1/m
        K = (Phi_X @ Phi_Y.T) / self.num_random_features

        if eval_gradient:
            # No hyperparameters, so just return gradient of shape (n_samples, n_samples, 0)
            return K, np.empty((X.shape[0], Y.shape[0], 0))

        return K

    def diag(self, X):
        # Get the diagonal elements
        Phi_X = np.cos(check_array(X) @ self.W.T)
        return np.sum(Phi_X**2, axis=1) / self.num_random_features

    def is_stationary(self):
        # Does not depend on absolute positions!
        return True
