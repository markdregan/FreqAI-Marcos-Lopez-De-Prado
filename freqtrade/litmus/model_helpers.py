# Helper functions for models
import numpy as np


class MergedModel:
    """Used for packaging multiple models together so that freqai can pickle the
    object for later usage

    Example usage:
        models = []
        models.append(model)
        model = MergedModel(models)"""

    def __init__(self, models):
        self.models = models
        self.classes_ = []
        for model in self.models:
            self.classes_.extend(list(model.classes_))

    def predict(self, X):
        results = []
        for model in self.models:
            results.append(model.predict(X))
        return np.hstack(results)

    def predict_proba(self, X):
        results = []
        for model in self.models:
            results.append(model.predict_proba(X))
        return np.hstack(results)
