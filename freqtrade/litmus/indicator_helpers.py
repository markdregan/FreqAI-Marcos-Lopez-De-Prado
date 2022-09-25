# Helper functions for trading indicators

import datetime
import logging

import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


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


def exclude_weak_features(model: str, pair: str, exclusion_threshold: float) -> np.array:
    """Identify weakest features from prior model feature selection routines
    and exclude these from future training

    :return list of column names that should be excluded for model + pair combo"""

    # Read trial + win data from sqlite
    connection_string = "sqlite:///litmus.sqlite"
    from_unix_date = datetime.datetime.utcnow() - datetime.timedelta(days=10)
    sql = f"""
        SELECT
            model, pair, feature_name,
            SUM(selected) as wins, SUM(1) AS trials
        FROM feature_selection_history
        WHERE model = '{model}'
        AND pair = '{pair}'
        AND train_time > '{from_unix_date}'
        GROUP BY 1,2,3"""

    feat_history = pd.read_sql_query(sql=sql, con=connection_string)

    # Generate random variate from distribution per feature based on prior inclusion / exclusion
    feat_history["rvs"] = stats.beta.rvs(1 + feat_history["wins"], 1 + feat_history["trials"])

    min_num_trials = 10
    excluded = feat_history.loc[
        (feat_history["rvs"] < exclusion_threshold) &
        (feat_history["trials"] > min_num_trials), "feature_name"].to_numpy()

    num_features_excluded = len(excluded)
    logger.info(f"Excluding {num_features_excluded} features based on prior selection history")

    return excluded
