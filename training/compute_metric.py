from transformers import EvalPrediction
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.metrics import classification_report
from scipy.special import softmax
from scipy.stats import pearsonr
import numpy as np


def compute_classification_metrics(eval_result: EvalPrediction):
    labels = eval_result.label_ids
    num_labels = len(np.unique(labels))
    if num_labels <= 2:
        y_pred = softmax(eval_result.predictions, axis=1)[:, 1]
        metrics = {
            "roc_auc_score": roc_auc_score(y_true=labels, y_score=y_pred),
            "average_precision_score": average_precision_score(
                y_true=labels, y_score=y_pred
            ),
        }

    else:
        y_pred = np.argmax(eval_result.predictions, axis=-1)
        metrics = {"mcc": matthews_corrcoef(labels, y_pred)}

        clf_metrics = classification_report(y_true=labels, y_pred=y_pred, output_dict=True, digits=4, target_names=[
                                            "class_{}".format(i) for i in range(num_labels)])
        for key, value in clf_metrics.items():
            if key == "accuracy":
                metrics[key] = value
            else:
                for sub_metric_name, v in value.items():
                    metrics[key + "_"+sub_metric_name] = v

    return metrics


def compute_regression_metrics(eval_result: EvalPrediction):
    labels = eval_result.label_ids
    y_pred = eval_result.predictions.flatten()
    r,p_value = pearsonr(y_pred, labels)
    metrics = {
            "pearsonr": r,
            "pearsonr_p_value":p_value,
            "rmse": mean_squared_error(y_true=labels, y_pred=y_pred, squared=False),
        }

    return metrics
