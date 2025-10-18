import numpy as np
from matplotlib import pyplot as plt

from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Cosine.NeuralCosineActivationKernel import NeuralCosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel
from Kernels.Tanh.TanhActivationKernel import TanhActivationKernel


########
# MEAN SQUARED ERROR CONVERGENCE: FINITE vs INFINITE COSINE KERNEL
########

def mean_square_error_from_kernels(x1, x2, m_values, num_trials=100):
    """Estimate MSE between finite and infinite cosine activation kernels."""

    # Instantiate infinite kernel
    infinite_kernel = TanhActivationKernel()

    # True analytic kernel value (CONFIRM WHATS GOING ON HERE PLEASE)
    k_true = infinite_kernel(x1.reshape(1, -1), x2.reshape(1, -1))[0, 0]

    mse_values = []
    for m in m_values:
        print(f"Estimating MSE for {m}")
        errors = []
        for _ in range(num_trials):
            finite_kernel = NeuralTanhActivationKernel(X=np.vstack([x1, x2]),
                                                         num_random_features=m)
            k_hat = finite_kernel(x1.reshape(1, -1), x2.reshape(1, -1))[0, 0]
            errors.append((k_hat - k_true) ** 2)
        mse_values.append(np.mean(errors))
    return np.array(mse_values)


# Two representative input vectors  TODO: xsin(x) dataset AND ACTUAL DATASET!!!
x1 = np.array([0.5, 0.8])
x2 = np.array([0.1, -0.4])

# Sample sizes
m_values = np.logspace(0, 6, 50, dtype=int)
mse_values = mean_square_error_from_kernels(x1, x2, m_values, num_trials=100)

# Plot convergence
plt.figure(figsize=(7, 5))
plt.loglog(m_values, mse_values, marker="o", label="Finite Cosine (Monte Carlo)")
plt.xlabel("Number of random features (m)")
plt.ylabel("Mean Squared Error")
plt.title("Convergence of Finite to Infinite Cosine Activation Kernel")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.savefig("output/mse_finite_vs_infinite_cosine_kernel.png")
plt.show()
