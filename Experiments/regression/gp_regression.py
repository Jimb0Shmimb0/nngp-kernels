import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Cosine.NeuralCosineActivationKernel import NeuralCosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel
from Experiments.datasets.datasets_utils import Concrete, Boston, Energy, Wine, Yacht
from Kernels.Tanh.TanhActivationKernel import TanhActivationKernel
from experiment_utils import evaluate_gp_predictions


DATASETS = {
    "Yacht": Yacht,
    "Boston": Boston,
    "Energy": Energy,
    "Concrete": Concrete,
    "Wine": Wine,
}

# Choose dataset
data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")
dataset = Yacht(out_dir=data_dir)
X_train, Y_train, X_test, Y_test = dataset.load_or_generate_data()

# Define data "unstandardising" function
unstandardise = lambda x : (x * dataset.Y_std) + dataset.Y_mean
Y_test_original = unstandardise(Y_test)

# Define noise variance constant alpha
ALPHA = 1e-3 # can be higher up to 0.1


#########
# RBF KERNEL
#########

# define kernel
rbf_kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# get GP regressor and fit to training datasets
rbf_gaussian_process = GaussianProcessRegressor(kernel=rbf_kernel, alpha=ALPHA, normalize_y=True)
rbf_gaussian_process.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
rbf_mean_prediction, rbf_std_prediction = rbf_gaussian_process.predict(X_test, return_std=True)

########
# COSINE ACTIVATION KERNEL
########

# define kernel
cosine_activation_kernel = CosineActivationKernel()

# get GP regressor and fit to training datasets
cos_gaussian_process = GaussianProcessRegressor(kernel=cosine_activation_kernel, alpha=ALPHA, normalize_y=True)
cos_gaussian_process.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
cos_mean_prediction, cos_std_prediction = cos_gaussian_process.predict(X_test, return_std=True)

########
# NEURAL NETWORK COSINE ACTIVATION KERNEL
########

# define kernel
finite_cosine_activation_kernel = NeuralCosineActivationKernel(X_train)

# get GP regressor and fit to training datasets
f_cos_gaussian_process = GaussianProcessRegressor(kernel=finite_cosine_activation_kernel, alpha=ALPHA, normalize_y=True)
f_cos_gaussian_process.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
neural_cos_mean_prediction, neural_cos_std_prediction = f_cos_gaussian_process.predict(X_test, return_std=True)

########
# TANH ACTIVATION KERNEL
########

# define kernel
tanh_activation_kernel = TanhActivationKernel()

# get GP regressor and fit to training datasets
tanh_gaussian_process = GaussianProcessRegressor(kernel=tanh_activation_kernel, alpha=ALPHA, normalize_y=True)
tanh_gaussian_process.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
tanh_mean_prediction, tanh_std_prediction = tanh_gaussian_process.predict(X_test, return_std=True)

########
# NEURAL NETWORK TANH ACTIVATION KERNEL
########

# define kernel
finite_tanh_activation_kernel = NeuralTanhActivationKernel(X_train)

# get GP regressor and fit to training datasets
neural_tanh_gaussian_process = GaussianProcessRegressor(kernel=finite_tanh_activation_kernel, alpha=ALPHA, normalize_y=True)
neural_tanh_gaussian_process.fit(X_train, Y_train)

# Get the mean and std prediction and plot the resulting gp regression
neural_tanh_mean_prediction, neural_tanh_std_prediction = neural_tanh_gaussian_process.predict(X_test, return_std=True)

K = finite_tanh_activation_kernel(X_train, X_train)


##########
# ANALYSIS
##########

evaluate_gp_predictions(
    "RBF Kernel",
    Y_test_original,
    unstandardise(rbf_mean_prediction),
)

evaluate_gp_predictions(
    "Cosine Activation Kernel",
    Y_test_original,
    unstandardise(cos_mean_prediction),
)

evaluate_gp_predictions(
    "Tanh Activation Kernel",
    Y_test_original,
    unstandardise(tanh_mean_prediction),
)

evaluate_gp_predictions(
    "Neural Cosine Activation Kernel",
    Y_test_original,
    unstandardise(neural_cos_mean_prediction),
)


evaluate_gp_predictions(
    "Neural Hyperbolic Tangent Activation Kernel",
    Y_test_original,
    unstandardise(neural_tanh_mean_prediction),
)
