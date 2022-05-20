# Helper functions for trading indicators

import numpy as np
import pandas as pd


def HA(open, high, low, close) -> pd.DataFrame:
    """Compute Heiken Ashi candles for any OHLC timeseries."""

    df_HA = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'is_green'],
                         index=range(0, len(open.index)))
    df_HA.loc[:, 'close'] = (open + high + low + close) / 4.0

    for i in range(0, len(open)):
        if i == 0:
            df_HA.at[i, 'open'] = ((open.iloc[i] + close.iloc[i]) / 2.0)
        else:
            df_HA.at[i, 'open'] = ((open.iloc[i - 1] + close.iloc[i - 1]) / 2.0)

    df_HA.loc[:, 'high'] = pd.concat([open, close, high], axis=1).max(axis=1)
    df_HA.loc[:, 'low'] = pd.concat([open, close, low], axis=1).min(axis=1)

    df_HA.loc[:, 'is_green'] = np.where(df_HA.loc[:, 'open'] < df_HA.loc[:, 'close'], 1, 0)

    return df_HA
