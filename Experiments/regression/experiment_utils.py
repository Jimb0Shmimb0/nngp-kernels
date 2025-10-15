import numpy as np

def mse(y_true, y_pred):
    """Mean squared Error"""
    return np.mean((y_true - y_pred) ** 2)
def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """Coefficient of determination r^2"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def evaluate_gp_predictions(model_name, y_true_raw, y_pred_mean, y_pred_std):
    """Compute and print standard GP regression metrics."""
    results = {
        "MSE": mse(y_true_raw, y_pred_mean),
        "RMSE": rmse(y_true_raw, y_pred_mean),
        "MAE": mae(y_true_raw, y_pred_mean),
        "R²": r2_score(y_true_raw, y_pred_mean),
    }

    print(f"\n=== {model_name} Performance ===")
    for k, v in results.items():
        print(f"{k:>8}: {v:.4f}")
    print("===============================")

    return results