# Helper functions for models
import logging

import numpy as np


logger = logging.getLogger(__name__)


class MergedModel:
    """Used for packaging multiple models together so that freqai can pickle the
    object for later usage

    Example usage:
        models = []
        models.append(model)
        model = MergedModel(models)"""

    def __init__(self, model_list, model_type_list, model_features_list):
        self.model_list = model_list
        self.classes_ = []
        for model in self.model_list:
            self.classes_.extend(list(model.classes_))
        self.model_type_list = model_type_list
        self.model_features_list = model_features_list

    def filter_features(self, X, model_features):
        if len(model_features) > 0:
            X = X.loc[:, model_features]

        return X

    def predict(self, X):
        results = []

        for i, model in enumerate(self.model_list):
            # Filter columns to what model expects
            X_temp = self.filter_features(X, self.model_features_list[i])

            # Sklearn and Catboost logloss needs reshaping
            results.append(model.predict(X_temp).reshape(-1, 1))

        return np.hstack(results)

    def predict_proba(self, X):
        results = []
        for i, model in enumerate(self.model_list):
            # Filter columns to what model expects
            X_temp = self.filter_features(X, self.model_features_list[i])

            # Add column vectors of probabilities to results list
            results.append(model.predict_proba(X_temp))

        return np.hstack(results)


def threshold_from_optimum_f1(precision, recall, thresholds, beta):
    """Compute the threshold cut off that aligns with the max F1 score"""
    numerator = (1 + beta ** 2) * recall * precision
    denom = recall + (beta ** 2 * precision)
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom),
                          where=(denom != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]

    return max_f1, max_f1_thresh


def threshold_from_desired_precision(precision, recall, thresholds, desired_precision=0.9):
    """Compute the threshold that corresponds to a min model precision"""
    desired_precision_idx = np.argmax(precision >= desired_precision)

    return recall[desired_precision_idx], thresholds[desired_precision_idx]


def precision_from_desired_threshold(precision, recall, thresholds, desired_threshold=0.9):
    """Compute the threshold that corresponds to a min model precision"""
    desired_threshold_idx = np.argmax(thresholds >= desired_threshold)

    return precision[desired_threshold_idx]
