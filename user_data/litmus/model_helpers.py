# Helper functions for models
import numpy as np


class MergedModel:

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


def get_threshold(search_list, lookup_list, search_threshold):
    # Ensure lists are equal length
    search_list = search_list[:len(lookup_list)]
    comp = search_list > search_threshold
    if comp.sum() == 0:
        return 1
    idx = np.argmax(comp)
    return lookup_list[idx]
