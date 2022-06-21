# Helper functions for trading indicators

from ta import add_all_ta_features

import pandas as pd


def heikin_ashi(df):
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
