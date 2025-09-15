import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils.validation import check_array

class CosineActivationKernel(Kernel):
    """Custom kernel: exp(-0.5(||x1||^2 + ||x2||^2)) * cosh(x1^T x2)."""

    def __init__(self):
        # There are no tunable hyperparameters for now
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        X = check_array(X)
        if Y is None:
            Y = X
        else:
            Y = check_array(Y)

        # Compute the norms
        X_norm = np.sum(X**2, axis=1)[:, np.newaxis]
        Y_norm = np.sum(Y**2, axis=1)[np.newaxis, :]

        # Compute the inner product. Transpose Y first
        X_dot_Y = np.dot(X, Y.T)

        # Then define the kernel matrix
        K = np.exp(-0.5 * (X_norm + Y_norm)) * np.cosh(X_dot_Y)

        if eval_gradient:
            # No hyperparameters → return gradient of shape (n_samples, n_samples, 0)
            return K, np.empty((X.shape[0], Y.shape[0], 0))

        return K

    def diag(self, X):
        # Diagonal entries K(x,x)
        return np.ones(X.shape[0])

    def is_stationary(self):
        # Depends on absolute positions (via norms), so not stationary
        return False
