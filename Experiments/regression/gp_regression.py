import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel
from Experiments.data_utils.datasets import Concrete, Boston, Energy, Kin8nm, Naval, Power, Protein, Wine, Yacht
import matplotlib.pyplot as plt

# Dataset: Concrete
data_dir = os.path.join(os.path.dirname(os.getcwd()), "data_utils")
dataset = Concrete(out_dir=data_dir)
X_train, Y_train, X_test, Y_test = dataset.load_or_generate_data()

# define kernel
kernel = CosineActivationKernel()

# get GP regressor and fit to training data_utils
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
mean_prediction, std_prediction = gp.predict(X_test, return_std=True) # alpha=noise_std**2

# Unstandardise (convert back to original units)
unstandardise = lambda x : (x * dataset.Y_std) + dataset.Y_mean
Y_pred_original = unstandardise(mean_prediction)
Y_test_original = unstandardise(Y_test)

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(Y_test_original, Y_pred_original, s=20, alpha=0.6, edgecolor="k")
plt.plot(
    [Y_test_original.min(), Y_test_original.max()],
    [Y_test_original.min(), Y_test_original.max()],
    "r--", lw=2, label="Ideal Fit"
)
plt.title(f"Cosine Activation GP Regression — {dataset.name.capitalize()} Dataset")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# TODO: Formalise tests. Move datasets to the datasets package. Fix the finite tanh kernel!!!!!!
# TODO: Start writing up notes properly. Draft the presentation notes!
