# Helper functions for trading indicators

from ta import add_all_ta_features

import numpy as np
import pandas as pd


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heiken Ashi candles for any OHLC timeseries."""

    heikin_ashi_df = pd.DataFrame(index=df.index.values, columns=['open', 'high', 'low', 'close'])

    heikin_ashi_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    for i in range(len(df)):
        if i == 0:
            heikin_ashi_df.iat[0, 0] = df['open'].iloc[0]
        else:
            heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i - 1, 0] + heikin_ashi_df.iat[
                i - 1, 3]) / 2

    heikin_ashi_df['high'] = heikin_ashi_df.loc[:, ['open', 'close']].join(df['high']).max(axis=1)

    heikin_ashi_df['low'] = heikin_ashi_df.loc[:, ['open', 'close']].join(df['low']).min(axis=1)

    return heikin_ashi_df


def add_all_ta_informative(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Add TA features and rename columns"""

    df_ta = add_all_ta_features(df, open="open", high="high", low="low",
                                close="close", volume="volume", fillna=False)
    df_ta.columns = [x + suffix for x in df_ta.columns]

    return df_ta


def add_single_ta_informative(df: pd.DataFrame, ta_method, suffix: str, col: str,
                              **kwargs) -> pd.DataFrame:

    df_ta = df.apply(lambda x: ta_method(x, **kwargs) if x.name == col else x)
    df_ta.columns = [x + suffix for x in df_ta.columns]

    return df_ta


def cusum_filter(df: pd.DataFrame, threshold_coeff: float) -> pd.DataFrame:
    """Sampling method using CUSUM given a threshold
        --------
        df: DataFrame must include 'colse' column
        threshold_coeff: percentage of daily volatility that defines sampling threshold"""

    # Daily volatility: 5mins x 288 = 1day
    df = daily_volatility(df, shift=288, lookback=50)

    df['entry_trigger'] = False
    df['cusum_pos_threshold'] = df['daily_volatility'] * threshold_coeff
    df['cusum_neg_threshold'] = df['daily_volatility'] * threshold_coeff * -1
    df['cusum_s_neg'] = 0

    s_pos = 0.0
    s_neg = 0.0

    # log returns
    diff = np.log(df['close']).diff()

    for i in diff.index:
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        # Track cusum variables for plotting
        df.loc[i, 'cusum_s_pos'] = s_pos
        df.loc[i, 'cusum_s_neg'] = s_neg

        if s_neg < df.loc[i, 'cusum_neg_threshold']:
            s_neg = 0
            df.loc[i, 'entry_trigger'] = True

        elif s_pos > df.loc[i, 'cusum_pos_threshold']:
            s_pos = 0
            df.loc[i, 'entry_trigger'] = True

    return df


def daily_volatility(close: pd.DataFrame, shift: int, lookback: int):
    """Compute daily volatility of price series
        --------
        dataframe: must contain column for close
        shift: number of candles to shift one day
        lookback: period over which ema averaging will be computed over
        """

    log_returns_daily = np.log(close / close.shift(shift))
    daily_volatility = log_returns_daily.ewm(span=lookback).std(ddof=0)

    return daily_volatility


def tripple_barrier(df_col, upper_pct, lower_pct):
    """Return label associated with first time crossing of vertical, upper or lower barrier"""

    initial_value = df_col.iat[0]
    upper_threshold = initial_value * (1 + upper_pct)
    lower_threshold = initial_value * (1 - lower_pct)

    # Get index position of first time upper & lower are crossed
    upper_idx = (df_col > upper_threshold).argmax() if (df_col > upper_threshold).any() else 999
    lower_idx = (df_col < lower_threshold).argmax() if (df_col < lower_threshold).any() else 999

    # Based on first crossing, assign appropriate label
    barrier = np.where(upper_idx < lower_idx, 1, np.NaN)
    barrier = np.where(lower_idx < upper_idx, -1, barrier)
    barrier = np.where(np.isnan(barrier), 0, barrier)

    return barrier


def max_extreme_value(df_col):
    """Identify max extreme value over future window and return value"""
    initial_value = df_col.iat[0]
    df_norm = df_col - initial_value

    # Get index position of first time upper & lower are crossed
    max_idx = df_norm.abs().argmax()

    return df_col.values[max_idx]
