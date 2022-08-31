from scipy.signal import find_peaks

import logging
import numba as nb
import numpy as np
import pandas as pd
import time

logger = logging.getLogger(__name__)


@nb.njit(fastmath=True, parallel=True, cache=True)
def minmax_growth(ts):
    ts = ts.astype(np.float32)
    res = np.empty((ts.shape[0], ts.shape[0]), dtype=ts.dtype)
    for i in nb.prange(ts.shape[0]):
        for j in range(ts.shape[0]):
            r = (ts[j] - ts[i]) / ts[i]
            res[i, j] = r
    return res


def minmax_growth_np(ts):
    """Just for testing without numba... very slow for large ts"""
    ts = ts.astype(np.float32)
    res = np.empty((ts.shape[0], ts.shape[0]), dtype=ts.dtype)
    for i in range(ts.shape[0]):
        for j in range(ts.shape[0]):
            r = (ts[j] - ts[i]) / ts[i]
            res[i, j] = r
    return res


def entry_exit_labeler(close_df, min_growth, max_duration):
    """Label timeseries for good entries and exits at peaks"""

    logger.info(f"Labeling for {len(close_df)} samples")
    logger.info(f"Labeling settings: min_growth {min_growth}, max_duration {max_duration}")
    start_time = time.time()

    close = close_df.values

    # Find all peaks and valleys
    exit_idx = find_peaks(close)[0]
    entry_idx = find_peaks(-close)[0]

    # Mask valleys & peaks
    min_mask = np.zeros(close.shape, dtype=bool)
    min_mask[entry_idx] = True
    max_mask = np.zeros(close.shape, dtype=bool)
    max_mask[exit_idx] = True

    # Compute distance matrix between all close points
    dist_matrix = minmax_growth_np(close)

    # Scope dist_matrix to distances > min_threshold
    growth_mask = (dist_matrix - min_growth) > 0
    dist_matrix = dist_matrix * growth_mask

    # Limit to max_duration & remove lower triangle
    dist_matrix = np.tril(dist_matrix, k=max_duration)
    dist_matrix = np.triu(dist_matrix)

    # Scope to only entries & get idx
    dist_matrix = dist_matrix * min_mask.reshape(-1, 1)
    good_entry_idx, _ = np.where(dist_matrix > 0)

    # Scope to entries & exits & find idx
    dist_matrix = dist_matrix * max_mask
    _, good_exit_idx = np.where(dist_matrix > 0)

    # Populate DataFrame and return
    target_name = "&target"
    label_names = ["good_entry", "good_exit", "bad_entry", "bad_exit", "transition"]
    df = pd.DataFrame(columns=[target_name], index=close_df.index)

    # Bad entries
    df.loc[[i for i in entry_idx if i not in good_entry_idx], target_name] = label_names[2]
    # Good entries
    df.loc[good_entry_idx, target_name] = label_names[0]
    # Bad exit
    df.loc[[i for i in exit_idx if i not in good_exit_idx], target_name] = label_names[3]
    # Good exit
    df.loc[good_exit_idx, target_name] = label_names[1]
    # The rest
    df.fillna(value=label_names[4], inplace=True)

    end_time = time.time() - start_time
    logger.info(f"Time taken to label data: {end_time} seconds")

    return df
