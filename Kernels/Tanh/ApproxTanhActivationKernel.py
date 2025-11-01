import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils.validation import check_array

class TanhActivationKernel(Kernel):
    """
    Kernel definition: (2/π) * arcsin( (2b^2 x^top y) / sqrt((1 + 2b^2|x|^2)(1 + 2b^2|y|^2)) )

    Reference: #TODO: PUT IN THE NOTES!!!!
      - Williams, C. K. I. (1997). "Computing with Infinite Networks".
    """

    def __init__(self, b=np.sqrt(np.pi)/2):
        self.b = b  # scaling factor mapping tanh roughly equal to erf(bx)

    def __call__(self, X, Y=None, eval_gradient=False):
        # Check X and Y are arrays
        X = check_array(X)
        if Y is None:
            Y = X
        else:
            Y = check_array(Y)

        # Compute the norms
        X_norm = np.sum(X ** 2, axis=1)[:, np.newaxis]
        Y_norm = np.sum(Y ** 2, axis=1)[np.newaxis, :]

        # Compute the inner product. Transpose Y first
        X_dot_Y = np.dot(X, Y.T)

        # Compute normalized correlation argument inside arcsin
        numerator = 2 * (self.b ** 2) * X_dot_Y
        denominator = np.sqrt((1 + 2 * (self.b ** 2) * X_norm) * (1 + 2 * (self.b ** 2) * Y_norm))
        frac = np.clip(numerator / denominator, -1.0, 1.0)  # keep the fraction numerically safe to pass into arcsin

        # Kernel matrix
        K = (2 / np.pi) * np.arcsin(frac)

        if eval_gradient:
            # No learnable hyperparameters exposed for autodiff here.
            return K, np.empty((X.shape[0], Y.shape[0], 0))

        return K

    def diag(self, X):
        # Diagonal elements (2/pi) * arcsin((2b^2 |x|^2) / (1 + 2b^2 |x|^2))
        X = check_array(X)
        norms = np.sum(X ** 2, axis=1)
        frac = (2 * (self.b ** 2) * norms) / (1 + 2 * (self.b ** 2) * norms)
        safe_frac = np.clip(frac, -1.0, 1.0)
        return (2 / np.pi) * np.arcsin(safe_frac)

    def is_stationary(self):
        return False