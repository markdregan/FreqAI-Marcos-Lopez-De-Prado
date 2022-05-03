# Experiment with CUSUM filter for primary model triggering
# Author: markdregan@gmail.com

from freqtrade.strategy import IStrategy, merge_informative_pair
from joblib import load
from pandas import DataFrame
from pathlib import Path
from ta import add_all_ta_features

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa


def cusum_filter(df, threshold):

    df['cusum_trigger'] = False
    s_pos = 0
    s_neg = 0

    # log returns
    diff = np.log(df['close']).diff()

    for i in diff.index:
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold is True:
            s_neg = 0
            df.loc[i, 'cusum_trigger'] = True

        elif s_pos > threshold is True:
            s_pos = 0
            df.loc[i, 'cusum_trigger'] = True

    return df['cusum_trigger']


class CUSUMPrimary(IStrategy):

    minimal_roi = {"0": 0.03, "600": -1}

    stoploss = -0.03

    plot_config = {
        "main_plot": {
            "SMA": {
                "color": "red"
            },
        },
        "subplots": {
            "RSI": {
                "RSI": {
                    "color": "green"
                },
                "RSI_SMA": {
                    "color": "purple"
                },
                "RSI_BuyThreshold": {
                    "color": "grey"
                },
            },
            "Growing SMA": {
                "growing_SMA": {
                    "color": "grey"
                },
            },
            "Sell Ind": {
                "stoch_sell_cross": {
                    "color": "red"
                },
            },
        },
    }

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        # Add additional informative pairs
        informative_pairs += [
            ("BTC/USDT", "30m"),
        ]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict, cusum: bool = True) -> DataFrame:
        # Meta model indicators
        # Apply TA indicators to the pair dataframe
        dataframe = add_all_ta_features(dataframe,
                                        open="open",
                                        high="high",
                                        low="low",
                                        close="close",
                                        volume="volume",
                                        fillna=True)

        # Get lower res data for pair in question
        dataframe_1d = self.dp.get_pair_dataframe(pair=metadata['pair'],
                                                  timeframe='1d')
        dataframe_1d = add_all_ta_features(dataframe_1d,
                                           open="open",
                                           high="high",
                                           low="low",
                                           close="close",
                                           volume="volume",
                                           fillna=False)

        # Add BTC 1h informative pair
        dataframe_btc_30m = self.dp.get_pair_dataframe(pair='BTC/USDT',
                                                       timeframe='30m')
        dataframe_btc_30m = add_all_ta_features(dataframe_btc_30m,
                                                open="open",
                                                high="high",
                                                low="low",
                                                close="close",
                                                volume="volume",
                                                fillna=False)
        skip_columns = ['date', 'pair']
        dataframe_btc_30m.columns = [
            x if x in skip_columns else x + '_btc'
            for x in dataframe_btc_30m.columns
        ]

        # Merge Informative Pairs
        dataframe = merge_informative_pair(dataframe,
                                           dataframe_1d,
                                           self.timeframe,
                                           '1d',
                                           ffill=True)
        dataframe = merge_informative_pair(dataframe,
                                           dataframe_btc_30m,
                                           self.timeframe,
                                           '30m',
                                           ffill=True)

        # Add reference to pair so ML model can generate feature for prediction
        dataframe['pair_copy'] = metadata['pair']

        # Sample trades using CUSUM filter (flag to disable for CUSUMMeta)
        if cusum is True:
            dataframe['log_returns_diff'] = np.log(dataframe['close']).diff()
            dataframe['cusum_trigger'] = cusum_filter(dataframe, threshold=0.03)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame,
                           metadata: dict) -> DataFrame:
        """
        Buy trend indicator when buy_trigger crosses from 0 to 1
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with buy column
        """

        dataframe.loc[
            dataframe['cusum_trigger']
            & (dataframe['volume'] > 0),
            ['buy', 'buy_tag']] = (1, 'primary_buy_trigger')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Disable populate sell
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with sell column
        """

        return dataframe


class CUSUMMeta(CUSUMPrimary):
    """Subclass of Primary that combines both primary strategy and
    secondary meta model. The main extensions are to a) load the ML model,
    b) make predictions and add to data frame & c) make buy_trend decisions
    based on primary strategy signals & meta model."""

    def __init__(self, *args, **kwargs):
        """Load the ML MetaModel so we can make predictions. Need to ensure
        __init()__ from super class is executed."""

        # Ensure primary strategy __init__ loads
        super().__init__(*args, **kwargs)

        # Load ML model & config details when strategy is instantiated
        filename = 'VulcanMeta'
        filepath = Path('user_data', 'meta_model', filename)
        model_and_config = load(open(filepath, 'rb'))
        self.clf = model_and_config['model']
        self.model_threshold = model_and_config['model_threshold']
        self.model_X_features = model_and_config['X_features']

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict, cusum: bool = False) -> DataFrame:
        """Extend the populate_indicators method from super class to add prediction
        probabilities from the meta model."""

        # Extend the super class to add additional signals
        dataframe = super().populate_indicators(dataframe, metadata, cusum=False)

        # Add predictions from the meta model to dataframe
        dataframe['meta_model_proba'] = self.clf.predict_proba(
            dataframe.loc[:, self.model_X_features])[:, 1]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame,
                           metadata: dict) -> DataFrame:
        """Add meta model prediction to primary model buy trend.
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with buy column"""

        dataframe.loc[
            ((qtpylib.crossed_above(dataframe['buy_trigger'], dataframe['buy_trigger_threshold']))
             & (dataframe['meta_model_proba'] > self.model_threshold)
             & (dataframe['volume'] > 0)  # Important for backtesting
             ), ['buy', 'buy_tag']] = (1, 'meta_buy_trigger')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Vulcan sell strategy
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with sell column
        """

        dataframe.loc[((dataframe["stoch_sell_cross"] is True) |
                       (dataframe["lower_low"] is True)),
                      ['sell', 'sell_tag']] = (1, 'sell_trigger')

        return dataframe
