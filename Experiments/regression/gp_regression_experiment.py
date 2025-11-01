import os
import time
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from Experiments.regression.constants import SEED, NUM_TRIALS
from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Cosine.NeuralCosineActivationKernel import NeuralCosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel
from Experiments.datasets.datasets_utils import Concrete, Boston, Energy, Wine, Yacht
from Kernels.Tanh.TanhActivationKernel import TanhActivationKernel
from experiment_utils import evaluate_gp_predictions, fit_and_predict_gp, rmse

# Choose dataset (One of Yacht(...), Boston(...), Energy(...), Concrete(...) or Wine(...))
dataset = Wine(out_dir=os.path.join(os.path.dirname(os.getcwd()), "datasets"))






# Load dataset and set RNG
X_train, Y_train, X_test, Y_test = dataset.load_or_generate_data()
RNG = np.random.default_rng(SEED)
start = time.time() # Timer

# Define data "unstandardising" function
unstandardise = lambda x : (x * dataset.Y_std) + dataset.Y_mean
Y_test_original = unstandardise(Y_test)

########
# GP REGRESSION EXPERIMENT
########

rmse_results = {"RBF": [], "Cos": [], "NN Cos": [], "Tanh": [], "NN Tanh": []}

def fit_and_predict_gps(rng):
    m_preds = {"RBF": None, "Cos": None, "NN Cos": None, "Tanh": None, "NN Tanh": None}
    # Fit data and predict on each GP using
    # RBF, Cosine activation, NN cosine activation, Tanh activation and NN Tanh activation kernels respectively
    m_preds["RBF"] = (
        fit_and_predict_gp(1 * RBF(length_scale=1.0, length_scale_bounds="fixed"),
        X_train, Y_train, X_test)
    )

    m_preds["Cos"] = (
        fit_and_predict_gp(
            CosineActivationKernel(),
            X_train, Y_train, X_test
        )
    )

    m_preds["NN Cos"] = (
        fit_and_predict_gp(
            NeuralCosineActivationKernel(X_train, random_state=rng.integers(0, 2**32)),
            X_train, Y_train, X_test
        )
    )

    m_preds["Tanh"] = (
        fit_and_predict_gp(
            TanhActivationKernel(),
            X_train, Y_train, X_test
        )
    )

    m_preds["NN Tanh"] = (
        fit_and_predict_gp(
            NeuralTanhActivationKernel(X_train, random_state=rng.integers(0, 2**32)),
            X_train, Y_train, X_test)
    )

    # Calculate the RMSE with the true mean and each predicted mean
    for kernel, mean in m_preds.items():
        true_mean = Y_test_original
        pred_mean = unstandardise(mean)
        rmse_results[kernel].append(rmse(true_mean, pred_mean))

# Main experiment loop. Run the experiment num_trials times
for _ in range(NUM_TRIALS):
    rng = np.random.default_rng(RNG.integers(0, 2**32))
    fit_and_predict_gps(rng)

# Evaluate gp predictions
evaluate_gp_predictions(rmse_results)

end = time.time()
print(f"Time elapsed: {end - start}")