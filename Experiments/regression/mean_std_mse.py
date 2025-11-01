import matplotlib.pyplot as plt
import numpy as np
import glob

from Experiments.regression.constants import MSE_NUM_M_VALUES

########
# MEAN SQUARED ERROR CONVERGENCE EXPERIMENT (ALL DATASETS)
########

# Set directory prefix (cos or tanh)
prefix = "tanh"
pattern = f"mse_array_output/{prefix}/*.txt"

m_values = np.logspace(0, 6, MSE_NUM_M_VALUES, dtype=int)

def mean_std_mse(pattern):
    arrays = []
    for filename in glob.glob(pattern):
        with open(filename, "r") as f:
            # Read file, get rid of newlines and brackets
            text = f.read()
            text = text.replace("[", "").replace("]", "").replace("\n", "")
        # Parse data
        data = np.fromstring(text, sep=' ')
        arrays.append(data)

    # Stack to shape (num_files, 50)
    stacked = np.vstack(arrays)
    # Compute mean and std
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)

    return mean, std

mean, std = mean_std_mse(pattern)

# Plot mean std mse
plt.figure(figsize=(7, 5))
plt.loglog(m_values, mean, marker="o", label="Mean MSE")
plt.fill_between(m_values, mean - std, mean + std, alpha=0.3, label='+- 1 Std') # Alpha = opacity
# plt.title("MSE Convergence of the Neural Cosine Activation Kernel towards its finite counterpart")
plt.xlabel("Number of random features (m)")
plt.ylabel("Mean Squared Error")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.savefig("mse_output/m_std_mse.png")
plt.show()