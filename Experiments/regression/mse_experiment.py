import os
import numpy as np

from Experiments.regression.constants import MSE_SEED, MSE_NUM_PAIRS, MSE_NUM_TRIALS, MSE_NUM_M_VALUES
from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Cosine.NeuralCosineActivationKernel import NeuralCosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel
from Experiments.datasets.datasets_utils import Concrete, Boston, Energy, Wine, Yacht
from Kernels.Tanh.ApproxTanhActivationKernel import TanhActivationKernel
import matplotlib.pyplot as plt
import time

########
# MEAN SQUARED ERROR CONVERGENCE EXPERIMENT (1 DATASET AT A TIME)
########

# Choose dataset (One of Yacht(...), Boston(...), Energy(...), Concrete(...) or Wine(...))
dataset = Yacht(out_dir=os.path.join(os.path.dirname(os.getcwd()), "datasets"))

# Set the kernel classes
infinite_kernel_class = CosineActivationKernel
finite_kernel_class = NeuralCosineActivationKernel





# Load dataset and set RNG
X_train, Y_train, X_test, Y_test = dataset.load_or_generate_data()
rng = np.random.RandomState(MSE_SEED)
start = time.time() # Start timer

def mean_square_error_from_kernels(X, m_values=MSE_NUM_M_VALUES, num_pairs=MSE_NUM_PAIRS, num_trials=MSE_NUM_TRIALS):
    infinite_kernel = infinite_kernel_class()
    mse_values = []

    # For each value of m
    for m in m_values:
        print(f"Estimating MSE for m = {m}")
        errors = []
        # Get a set number of random pairs, calculate the true k(x1, x2)
        # estimate k(x1, x2) using the finite kernel for the set number of trials, and compute the mean MSE
        for _ in range(num_pairs):
            i, j = rng.choice(len(X), 2, replace=False)
            x1, x2 = X[i:i+1], X[j:j+1]

            k_true = infinite_kernel(x1, x2)[0, 0]
            for _ in range(num_trials):
                finite_kernel = finite_kernel_class(X=np.vstack([x1, x2]),
                                                             num_random_features=m)
                k_hat = finite_kernel(x1, x2)[0, 0]
                errors.append((k_hat - k_true) ** 2)
        mse_values.append(np.mean(errors))
    return np.array(mse_values)


# Sample sizes
m_values = np.logspace(0, 6, MSE_NUM_M_VALUES, dtype=int)
mse_values = mean_square_error_from_kernels(X_train, m_values)
print(mse_values)

# Plot convergence
plt.figure(figsize=(7, 5))
plt.loglog(m_values, mse_values, marker="o", label="MSE")
plt.xlabel("Number of random features (m)")
plt.ylabel("Mean Squared Error")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.savefig("mse_output/mse.png")
plt.show()

end = time.time()
print(f"Time elapsed: {end - start}")