import os
import numpy as np
from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Cosine.NeuralCosineActivationKernel import NeuralCosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel
from Experiments.datasets.datasets_utils import Concrete, Boston, Energy, Wine, Yacht
from Kernels.Tanh.TanhActivationKernel import TanhActivationKernel
import matplotlib.pyplot as plt
import time

start = time.time()

# Choose dataset
data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")
dataset = Yacht(out_dir=data_dir)
X_train, Y_train, X_test, Y_test = dataset.load_or_generate_data()

# Define data "unstandardising" function
unstandardise = lambda x : (x * dataset.Y_std) + dataset.Y_mean
Y_test_original = unstandardise(Y_test)

rng = np.random.RandomState(42)

########
# MEAN SQUARED ERROR CONVERGENCE
########

def mean_square_error_from_kernels(X, m_values, num_pairs=50, num_trials=100):
    infinite_kernel = TanhActivationKernel()
    mse_values = []

    for m in m_values:
        print(f"Estimating MSE for m = {m}")
        errors = []
        for _ in range(num_pairs):
            i, j = rng.choice(len(X), 2, replace=False)
            x1, x2 = X[i:i+1], X[j:j+1]

            k_true = infinite_kernel(x1, x2)[0, 0]
            for _ in range(num_trials):
                finite_kernel = NeuralTanhActivationKernel(X=np.vstack([x1, x2]),
                                                             num_random_features=m)
                k_hat = finite_kernel(x1, x2)[0, 0]
                errors.append((k_hat - k_true) ** 2)
        mse_values.append(np.mean(errors))
    return np.array(mse_values)


# Sample sizes
m_values = np.logspace(0, 6, 50, dtype=int)
mse_values = mean_square_error_from_kernels(X_train, m_values)
print(mse_values)

# Plot convergence
plt.figure(figsize=(7, 5))
plt.loglog(m_values, mse_values, marker="o", label="MSE")
plt.xlabel("Number of random features (m)")
plt.ylabel("Mean Squared Error")
plt.title("Convergence of NN Cosine Activation to Infinite Cosine Activation Kernel")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.savefig("output/mse_finite_vs_infinite_cosine_kernel.png")
plt.show()

end = time.time()

print(f"Time elapsed: {end - start}")