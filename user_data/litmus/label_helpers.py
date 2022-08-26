from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def cdist_growth(a, b):
    """Compute simple growth rate from a to b"""
    diff = b - a
    return np.divide(diff, a)


def entry_exit_labeler(close_df, direction, min_growth, max_duration):
    """Label timeseries for good entries and exits at peaks"""

    logger.info(f"Labeling for {direction} trades: Growth {min_growth}, Duration {max_duration}")
    close = close_df.values
    if direction == "short":
        close = close * -1

    # Find all peaks and valleys
    max_idx = find_peaks(close)[0]
    min_idx = find_peaks(-close)[0]

    # Mask valleys & peaks
    min_mask = np.zeros(close.shape, dtype=bool)
    min_mask[min_idx] = True
    max_mask = np.zeros(close.shape, dtype=bool)
    max_mask[max_idx] = True

    # Compute joint max x min joint mask
    if direction == "long":
        minmax_mask = np.outer(min_mask, max_mask)
    elif direction == "short":
        minmax_mask = np.outer(max_mask, min_mask)

    # Compute distance matrix between all close points
    dist_matrix = cdist(close.reshape(-1, 1), close.reshape(-1, 1), metric=cdist_growth)

    # Scope dist_matrix to distances > min_threshold
    growth_mask = (dist_matrix - min_growth) > 0
    dist_matrix = dist_matrix * growth_mask

    # Limit to max_duration & remove lower triangle
    dist_matrix = np.tril(dist_matrix, k=max_duration)
    dist_matrix = np.triu(dist_matrix)

    # Scope to only peak & valley points
    dist_matrix = minmax_mask * dist_matrix
    dist_matrix

    # Get idx for non zero distances
    good_entry_idx, good_exit_idx = np.where(dist_matrix > 0)

    # Populate DataFrame and return
    if direction == "long":
        col_names = ["&good_long_entry", "&good_long_exit",
                     "&fake_long_entry", "&fake_long_exit"]
    elif direction == "short":
        col_names = ["&good_short_entry", "&good_short_exit",
                     "&fake_short_entry", "&fake_short_exit"]
    df = pd.DataFrame(columns=col_names, index=close_df.index)
    df[col_names] = "No"

    # Entries & Exits
    if direction == "long":
        entry_idx = min_idx
        exit_idx = max_idx
    elif direction == "short":
        entry_idx = max_idx
        exit_idx = min_idx

    df.loc[[i for i in entry_idx if i not in good_entry_idx], col_names[2]] = "Yes"
    df.loc[good_entry_idx, col_names[0]] = "Yes"
    df.loc[[i for i in exit_idx if i not in good_exit_idx], col_names[3]] = "Yes"
    df.loc[good_exit_idx, col_names[1]] = "Yes"

    return df
