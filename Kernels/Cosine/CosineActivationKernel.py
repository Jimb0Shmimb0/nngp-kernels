import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils.validation import check_array

class CosineActivationKernel(Kernel):
    """Kernel definition: exp(-0.5(||x1||^2 + ||x2||^2)) * cosh(x1^T x2)."""

    def __init__(self):
        # There are no tunable hyperparameters. Just pass
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        # Check X and Y are arrays
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
            # No hyperparameters, so just return gradient of shape (n_samples, n_samples, 0)
            return K, np.empty((X.shape[0], Y.shape[0], 0))

        return K

    def diag(self, X):
        # Get the diagonal elements
        norms = np.sum(X ** 2, axis=1)
        return np.exp(-norms) * np.cosh(norms)

    def is_stationary(self):
        # Depends on absolute positions through norms, so not stationary
        return False
