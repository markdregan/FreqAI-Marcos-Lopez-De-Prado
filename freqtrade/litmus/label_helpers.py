import logging

import pandas as pd


logger = logging.getLogger(__name__)


def tripple_barrier(df_col, upper_pct, lower_pct, result):
    """Return label associated with first time crossing of vertical, upper or lower barrier

    upper_pct: % above close first cancle
    lower_pct: % below close first candle
    result: one of ["side", "value"]

    Example useage:
        window = 5
        params = {"upper_pct": 0.003, "lower_pct": 0.003}
        df["tripple_barrier"] = (
            df["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )
        tbm_map = {1: "upper", 0: "vertical", -1: "lower"}
        df["&tbm_target"] = df["tripple_barrier"].map(tbm_map)
    """

    initial_value = df_col.iat[0]
    upper_threshold = initial_value * (1 + upper_pct)
    lower_threshold = initial_value * (1 - lower_pct)

    # Get index position of first time upper & lower are crossed
    upper_idx = (df_col > upper_threshold).argmax() if (df_col > upper_threshold).any() else 99999
    lower_idx = (df_col < lower_threshold).argmax() if (df_col < lower_threshold).any() else 99999

    # Based on first crossing, assign appropriate label
    if upper_idx < lower_idx:
        side = 1
        value = df_col.iloc[upper_idx]
    elif lower_idx < upper_idx:
        side = -1
        value = df_col.iloc[lower_idx]
    else:
        side = 0
        value = df_col.iloc[-1]

    if result == "side":
        return side
    elif result == "value":
        return value


def max_extreme_value(df_col):
    """Identify max extreme value over future window and return value"""

    initial_value = df_col.iat[0]
    df_norm = df_col - initial_value

    # Get index position of first time upper & lower are crossed
    idx = df_norm.abs().argmax()

    return df_col.values[idx]


def nearby_extremes(df, threshold: float, forward_pass: bool, reverse_pass: bool):
    """Identify values beside peaks/valleys that are within a
    threshold distance and re-label them"""

    df.rename(columns={df.columns[1]: "raw_peaks"}, inplace=True)
    df_holder = []
    df_holder.append(df["raw_peaks"])

    # Forward Pass
    if forward_pass:
        forward_df = (pd.concat(
            [
                df,
                df.mask(df["raw_peaks"] == 0).fillna(
                    method="ffill").rename(columns=lambda n: "prev_" + n),
            ],
            axis=1
        )
                      .eval("within_threshold = abs(close-prev_close)/prev_close < @threshold")
                      .eval("mask = within_threshold and raw_peaks == 0")
                      .eval("is_new_extreme = within_threshold.mask(mask).fillna(method='ffill')")
                      .eval("forward_extremes = prev_raw_peaks.where(is_new_extreme).fillna(0)")
                      ["forward_extremes"]
                      .astype(int))
        df_holder.append(forward_df)

    # Reverse Pass
    if reverse_pass:
        df = df.sort_index(ascending=False)
        reverse_df = (pd.concat(
            [
                df,
                df.mask(df["raw_peaks"] == 0).fillna(
                    method="ffill").rename(columns=lambda n: "prev_" + n),
            ],
            axis=1
        )
                      .eval("within_threshold = abs(close-prev_close)/prev_close < @threshold")
                      .eval("mask = within_threshold and raw_peaks == 0")
                      .eval("is_new_extreme = within_threshold.mask(mask).fillna(method='ffill')")
                      .eval("reverse_extremes = prev_raw_peaks.where(is_new_extreme).fillna(0)")
                      ["reverse_extremes"]
                      .astype(int)).sort_index()
        df_holder.append(reverse_df)

    # Merging
    merged_df = pd.concat(df_holder, axis=1)
    final_df = merged_df.sum(axis=1).clip(lower=-1, upper=1)

    return final_df
