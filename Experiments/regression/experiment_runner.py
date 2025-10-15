import os

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from Kernels.Cosine.CosineActivationKernel import CosineActivationKernel
from Kernels.Cosine.NeuralCosineActivationKernel import NeuralCosineActivationKernel
from Kernels.Tanh.NeuralTanhActivationKernel import NeuralTanhActivationKernel
from Experiments.datasets.datasets_utils import Concrete, Boston, Energy, Kin8nm, Naval, Power, Protein, Wine, Yacht
from experiment_utils import evaluate_gp_predictions
import pandas as pd

ALPHA = 1e-2

def init_rbf():
    return GaussianProcessRegressor(
        kernel=RBF(length_scale_bounds=(ALPHA, 1e3)),
        alpha=ALPHA,
        normalize_y=True,
        n_restarts_optimizer=5
    )

def init_cosine():
    return GaussianProcessRegressor(
        kernel=CosineActivationKernel(),
        alpha=ALPHA,
        normalize_y=True
    )

def init_neural_cosine(X_train):
    return GaussianProcessRegressor(
        kernel=NeuralCosineActivationKernel(X_train),
        alpha=ALPHA,
        normalize_y=True
    )

def init_neural_tanh(X_train):
    return GaussianProcessRegressor(
        kernel=NeuralTanhActivationKernel(X_train),
        alpha=ALPHA,
        normalize_y=True
    )

MODELS = {
    "RBF": init_rbf,
    "Cosine": init_cosine,
    "NeuralCosine": init_neural_cosine,
    "NeuralTanh": init_neural_tanh,
}

DATASETS = {
    "Boston": Boston,
    "Concrete": Concrete,
    "Energy": Energy,
    "Kin8nm": Kin8nm,
    "Naval": Naval,
    "Power": Power,
    "Protein": Protein,
    "Wine": Wine,
    "Yacht": Yacht,
}

results = []

for dataset_name, dataset_cls in DATASETS.items():
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")
    dataset = dataset_cls(out_dir=data_dir)
    X_train, Y_train, X_test, Y_test = dataset.load_or_generate_data()

    unstandardise = lambda x: (x * dataset.Y_std) + dataset.Y_mean
    Y_test_original = unstandardise(Y_test)

    print(f"\n\n --------------- Running on dataset: {dataset_name} --------------- ")
    for model_name, init in MODELS.items():
        print(f"\n--- {model_name} ---")

        gp = init(X_train) if "Neural" in model_name else init()

        gp.fit(X_train, Y_train)

        y_pred, y_std = gp.predict(X_test, return_std=True)
        y_pred_raw = unstandardise(y_pred)

        scores = evaluate_gp_predictions(model_name, Y_test_original, y_pred_raw)

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            **scores
        })


df = pd.DataFrame(results)
df = df[["dataset", "model", "mean squared error", "coefficient of cetermination"]]
print("\n--- Aggregate Results ---")
print(df.to_string(index=False))
df.to_csv("output/results.csv", index=False)


#TODO: Hook this up with mse.py and before running this, test gp_regression.py on ALL datasets.