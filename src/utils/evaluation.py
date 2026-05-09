import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    classification_report, roc_auc_score, confusion_matrix
)


def regression_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True)
    result = {"classification_report": report}
    if y_prob is not None:
        result["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return result
