import inspect

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Cosine.NeuralCosineActivationKernel import NeuralCosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel

f = lambda x: x * np.sin(x)

def plot_gaussian_process_regression(X, f, mean_prediction, std_prediction, kernel_type):
    string_rep = inspect.getsourcelines(f)[0][0].split(':')[1].strip()

    plt.close()
    plt.plot(X, f(X), label=f"{string_rep}", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title(f"Gaussian process regression on noise-free dataset \n "
                  f"using {kernel_type} kernel")
    plt.savefig(f'output/gp_reg_{kernel_type.lower().replace(" ", "_")}.png')

def generate_and_plot_data(X, f):
    # noise_std = 0.75
    y = np.squeeze(f(X)) # + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)
    # create Y by applying f(x), then flatten
    string_rep = inspect.getsourcelines(f)[0][0].split(':')[1].strip()

    plt.close()
    plt.plot(X, y, label=f"{string_rep}", linestyle="dotted")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("True generative process")
    plt.savefig(f'output/{string_rep}.png')

    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=6, replace=False)  # pick 6 out of [0, 1, 2, …, 999]
    return X[training_indices], y[training_indices]




#########
# Dataset generation
#########

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1) # 1000 evenly spaced points between 0 and 10. Shape: (1000x1)
X_train, y_train = generate_and_plot_data(X, f)

#########
# RBF KERNEL TEST
#########

# define kernel
rbf_kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# get GP regressor and fit to training datasets
rbf_gaussian_process = GaussianProcessRegressor(kernel=rbf_kernel, n_restarts_optimizer=9) # alpha=noise_std**2
rbf_gaussian_process.fit(X_train, y_train)

# Get the mean and std prediction and plot the resulting gp regression
mean_prediction, std_prediction = rbf_gaussian_process.predict(X, return_std=True)
plot_gaussian_process_regression(X, f, mean_prediction, std_prediction, "RBF")



########
# COSINE ACTIVATION KERNEL TEST
########

# define kernel
cosine_activation_kernel = CosineActivationKernel()

# get GP regressor and fit to training datasets
cos_gaussian_process = GaussianProcessRegressor(kernel=cosine_activation_kernel)
cos_gaussian_process.fit(X_train, y_train)

# Get the mean and std prediction and plot the resulting gp regression
mean_prediction, std_prediction = cos_gaussian_process.predict(X, return_std=True) # alpha=noise_std**2
plot_gaussian_process_regression(X, f, mean_prediction, std_prediction, "Cosine activation")




########
# FINITE COSINE ACTIVATION KERNEL TEST
########

# define kernel
finite_cosine_activation_kernel = NeuralCosineActivationKernel(X)

# get GP regressor and fit to training datasets
f_cos_gaussian_process = GaussianProcessRegressor(kernel=finite_cosine_activation_kernel)
f_cos_gaussian_process.fit(X_train, y_train)

# Get the mean and std prediction and plot the resulting gp regression
mean_prediction, std_prediction = f_cos_gaussian_process.predict(X, return_std=True) # alpha=noise_std**2
plot_gaussian_process_regression(X, f, mean_prediction, std_prediction, "Finite Cosine activation")


########
# FINITE TANH ACTIVATION KERNEL TEST
########

# TODO: Kernel is not returning a positive definite matrix. Fix please!
# define kernel
finite_tanh_activation_kernel = NeuralTanhActivationKernel(X)

# get GP regressor and fit to training datasets
f_tanh_gaussian_process = GaussianProcessRegressor(kernel=finite_tanh_activation_kernel, alpha=1e-5) # Too much noise. Fix
f_tanh_gaussian_process.fit(X_train, y_train)

# Get the mean and std prediction and plot the resulting gp regression
mean_prediction, std_prediction = f_tanh_gaussian_process.predict(X, return_std=True) # alpha=noise_std**2
plot_gaussian_process_regression(X, f, mean_prediction, std_prediction, "Finite Tanh activation")

