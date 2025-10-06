import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils.validation import check_array
from scipy.stats import logistic, multivariate_normal


class FiniteTanhActivationKernel(Kernel):
    """
    Finite-dimensional approximation for tanh activation kernel:
        k(x1, x2) = 4 * P(L <= Z, L' <= Z') - 1
    where L, L' \sim Logistic(0, 1/2) and (Z, Z') \sim N(0, Sigma)
    with Sigma = [[||x1||^2, x1^T x2],
              [x2^T x1, ||x2||^2]]

    Approximate using num_random_features samples.
    """

    def __init__(self, X, num_random_features=200, random_state=None):
        self.num_random_features = num_random_features
        self.random_state = random_state

        self.W = self.random_state.normal(
            loc=0.0,
            scale=1.0,
            size=(self.num_random_features, X.shape[1])
        ) # w_i \sim \mathcal{N}(0, I)

        # Note: If we want weights to be initialised during instantiation of this kernel, input X has to be passed in
        # W needs to match the shape of X.

    def _estimate_kernel(self, x1, x2):
        """ Estimate k(x1, x2) using the law of large numbers """

        # Compute covariance matrix
        cov = np.array([
            [np.dot(x1, x1), np.dot(x1, x2)],
            [np.dot(x2, x1), np.dot(x2, x2)]
        ])

        # Sample num_random_features amount of Z's and L's
        Z_samples = self.random_state.multivariate_normal(mean=[0, 0], cov=cov, size=self.num_random_features)
        L_samples = self.random_state.logistic(loc=0.0, scale=0.5, size=(self.num_random_features, 2))

        # Find probabilities
        probs = (L_samples[:, 0] <= Z_samples[:, 0]) & (L_samples[:, 1] <= Z_samples[:, 1])
        return 4.0 * np.mean(probs) - 1.0

    def __call__(self, X, Y=None, eval_gradient=False):
        # Check X and Y are arrays
        X = check_array(X)
        if Y is None:
            Y = X
        else:
            Y = check_array(Y)

        K = np.zeros((X.shape[0], Y.shape[0]))

        # Compute each element of the kernel matrix
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                K[i, j] = self._estimate_kernel(X[i], Y[j])

        if eval_gradient:
            return K, np.empty((X.shape[0], Y.shape[0], 0))
        return K

    def diag(self, X):
        """Return the diagonal entries."""
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True # Need to confirm. Why is it stationary?
