import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils.validation import check_array
from scipy.stats import logistic, multivariate_normal


class NeuralTanhActivationKernel(Kernel):
    """
    Finite-dimensional approximation for tanh activation kernel:
        k(x1, x2) = 4 * P(L <= Z, L' <= Z') - 1
    where L, L' sim Logistic(0, 1/2) and (Z, Z') sim N(0, Sigma)
    with Sigma = [[||x1||^2, x1^T x2],
              [x2^T x1, ||x2||^2]]

    Approximate using num_random_features samples.
    """

    def __init__(self, X, num_random_features=20000, random_state=42):
        self.num_random_features = num_random_features
        self.random_state = random_state

        self.W = np.random.RandomState(self.random_state).normal(
            loc=0.0,
            scale=1.0,
            size=(self.num_random_features, X.shape[1])
        )  # w_i sim mathcal{N}(0, I)

        self.L1 = np.random.RandomState(self.random_state).logistic(loc=0.0, scale=0.5,
                                                                    size=(self.num_random_features, 1))
        self.L2 = np.random.RandomState(self.random_state).logistic(loc=0.0, scale=0.5,
                                                                    size=(self.num_random_features, 1))

        # Note: If we want weights to be initialised during instantiation of this kernel, input X has to be passed in
        # W needs to match the shape of X.

        self.X = X

    def _estimate_kernel_matrix(self, X, Y):
        Z1 = (X @ self.W.T).T # self.num_features x N
        Z2 = (Y @ self.W.T).T # self.num_features x M

        indicator1 = (self.L1 <= Z1).astype(np.float32) # (self.num_features x N)
        indicator2 = (self.L2 <= Z2).astype(np.float32) # (self.num_features x M)

        K = indicator1.T @ indicator2/self.num_random_features # (N x M)

        if X.shape == Y.shape:
            K = 0.5 * (K + K.T)
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            eigenvalues = np.maximum(eigenvalues, 1e-4)
            K = (eigenvectors * eigenvalues) @ eigenvectors.T

        return 4.0 * K - 1.0

    def __call__(self, X, Y=None, eval_gradient=False):
        # Check X and Y are arrays
        X = check_array(X)
        if Y is None:
            Y = X
        else:
            Y = check_array(Y)

        K = self._estimate_kernel_matrix(X, Y)

        if eval_gradient:
            return K, np.empty((X.shape[0], Y.shape[0], 0))
        return K

    def diag(self, X):
        X = check_array(X)
        return np.diag(self._estimate_kernel_matrix(X,X))

    def is_stationary(self):
        return False