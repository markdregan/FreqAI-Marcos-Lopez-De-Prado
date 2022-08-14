# Helper functions for models
import numpy as np


def get_threshold(search_list, lookup_list, search_threshold):
    comp = search_list > search_threshold
    if comp.sum() == 0:
        return 1
    idx = np.argmax(comp)
    return lookup_list[idx]
