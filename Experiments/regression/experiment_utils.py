import numpy as np

def mse(y_true, y_pred):
    # Mean squared error
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    # Coefficient of determination
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def evaluate_gp_predictions(model_name, y_true_raw, y_pred_mean):
    # Print scores
    results = {
        "MSE": mse(y_true_raw, y_pred_mean),
        "R Squared": r2_score(y_true_raw, y_pred_mean),
    }

    print(f"\n=== {model_name} Performance ===")
    for k, v in results.items():
        print(f"{k:>8}: {v:.4f}")
    print("===============================")

    return results
