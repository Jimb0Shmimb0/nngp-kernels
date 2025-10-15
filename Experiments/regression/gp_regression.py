import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Cosine.NeuralCosineActivationKernel import NeuralCosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel
from Experiments.datasets.datasets_utils import Concrete, Boston, Energy, Kin8nm, Naval, Power, Protein, Wine, Yacht
import matplotlib.pyplot as plt

# Choose dataset
data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")
dataset = Concrete(out_dir=data_dir)
X_train, Y_train, X_test, Y_test = dataset.load_or_generate_data()

# Define data "unstandardising" function
unstandardise = lambda x : (x * dataset.Y_std) + dataset.Y_mean
Y_test_original = unstandardise(Y_test)

# Define noise variance constant alpha
alpha = 1e-4

########
# COSINE ACTIVATION KERNEL
########

# define kernel
cosine_activation_kernel = CosineActivationKernel()

# get GP regressor and fit to training datasets
cos_gaussian_process = GaussianProcessRegressor(kernel=cosine_activation_kernel, alpha=alpha)
cos_gaussian_process.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
cos_mean_prediction, cos_std_prediction = cos_gaussian_process.predict(X_test, return_std=True)

# Unstandardise the data
raw_Y_prediction_cos = unstandardise(cos_mean_prediction)


########
# NEURAL NETWORK COSINE ACTIVATION KERNEL
########

# define kernel
finite_cosine_activation_kernel = NeuralCosineActivationKernel(X_train)

# get GP regressor and fit to training datasets
f_cos_gaussian_process = GaussianProcessRegressor(kernel=finite_cosine_activation_kernel, alpha=alpha)
f_cos_gaussian_process.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
neural_cos_mean_prediction, neural_std_prediction = f_cos_gaussian_process.predict(X_test, return_std=True)

# Unstandardise the data
raw_Y_prediction_neural_cos = unstandardise(neural_cos_mean_prediction)


########
# NEURAL NETWORK TANH ACTIVATION KERNEL
########

# define kernel
finite_tanh_activation_kernel = NeuralTanhActivationKernel(X_train)

# get GP regressor and fit to training datasets
f_tanh_gaussian_process = GaussianProcessRegressor(kernel=finite_tanh_activation_kernel, alpha=alpha)
f_tanh_gaussian_process.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
neural_tanh_mean_prediction, neural_tanh_std_prediction = f_tanh_gaussian_process.predict(X_test, return_std=True)

# Unstandardise the data
raw_Y_prediction_neural_tanh = unstandardise(neural_tanh_mean_prediction)

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def nll(y_true, mu, std, eps=1e-9):
    """Negative log-likelihood under Gaussian predictive distribution."""
    var = np.maximum(std ** 2, eps)
    return np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * ((y_true - mu) ** 2) / var)


def coverage(y_true, mu, std, k=1.0):
    """Empirical coverage of k-sigma interval."""
    return np.mean(np.abs(y_true - mu) <= k * std)


def evaluate_gp_predictions(model_name, y_true_raw, y_pred_mean, y_pred_std):
    """Compute and print standard GP regression metrics."""
    results = {
        "RMSE": rmse(y_true_raw, y_pred_mean),
        "MAE": mae(y_true_raw, y_pred_mean),
        "R²": r2_score(y_true_raw, y_pred_mean),
        "NLL": nll(y_true_raw, y_pred_mean, y_pred_std),
        "COV@1σ": coverage(y_true_raw, y_pred_mean, y_pred_std, k=1),
        "COV@2σ": coverage(y_true_raw, y_pred_mean, y_pred_std, k=2),
    }

    print(f"\n=== {model_name} Performance ===")
    for k, v in results.items():
        print(f"{k:>8}: {v:.4f}")
    print("===============================")

    return results


evaluate_gp_predictions(
    "Cosine Activation Kernel",
    raw_Y_prediction_cos,
    cos_mean_prediction,
    cos_std_prediction
)

evaluate_gp_predictions(
    "Neural Cosine Activation Kernel",
    raw_Y_prediction_neural_cos,
    neural_cos_mean_prediction,
    cos_std_prediction
)

evaluate_gp_predictions(
    "Neural Hyperbolic Tangent Activation Kernel",
    raw_Y_prediction_neural_tanh,
    neural_tanh_mean_prediction,
    neural_tanh_std_prediction
)


# TODO: Formalise tests. Move datasets to the datasets package. Fix the finite tanh kernel!!!!!!
# TODO: Start writing up notes properly. Draft the presentation notes!
