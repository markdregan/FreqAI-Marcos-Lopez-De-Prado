import logging


logger = logging.getLogger(__name__)


def tripple_barrier(df_col, upper_pct, lower_pct):
    """Return label associated with first time crossing of vertical, upper or lower barrier

    Example useage:
        df["tripple_barrier"] = (
            df["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )
    """

    initial_value = df_col.iat[0]
    upper_threshold = initial_value * (1 + upper_pct)
    lower_threshold = initial_value * (1 - lower_pct)

    # Get index position of first time upper & lower are crossed
    upper_idx = (df_col > upper_threshold).argmax() if (df_col > upper_threshold).any() else 99999
    lower_idx = (df_col < lower_threshold).argmax() if (df_col < lower_threshold).any() else 99999

    # Based on first crossing, assign appropriate label
    if upper_idx < lower_idx:
        barrier = 1
    elif lower_idx < upper_idx:
        barrier = -1
    else:
        barrier = 0

    return barrier


def max_extreme_value(df_col):
    """Identify max extreme value over future window and return value"""

    initial_value = df_col.iat[0]
    df_norm = df_col - initial_value

    # Get index position of first time upper & lower are crossed
    idx = df_norm.abs().argmax()

    return df_col.values[idx]
