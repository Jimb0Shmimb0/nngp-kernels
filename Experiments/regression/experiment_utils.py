import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from Experiments.regression.constants import ALPHA

def fit_and_predict_gp(kernel, X_train, Y_train, X_test):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=ALPHA, normalize_y=True, optimizer=None)
    gp.fit(X_train, Y_train)
    mean = gp.predict(X_test)
    return mean

def rmse(y_true, y_pred):
    # Root mean squared error
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def evaluate_gp_predictions(rmse_results):
    # Print and return rmse results
    final_results = {"RBF": None, "Cos": None, "Tanh": None, "NN Cos": None, "NN Tanh": None}

    for kernel, results in rmse_results.items():
        rmse_mean = np.mean(results)
        rmse_std = np.std(results)

        final_results[kernel] = (rmse_mean, rmse_std)

        print(f"\n======== {kernel} Kernel Performance ========")
        print(f"RMSE mean: {rmse_mean:8.4f}, RMSE std:{rmse_std:8.4f}") # 8 chars wide, 4 dec. places. Right aligned
        print("=======================================")

    return final_results
