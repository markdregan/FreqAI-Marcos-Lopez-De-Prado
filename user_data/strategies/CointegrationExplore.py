# Author: markdregan@gmail.com

from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from ta import add_all_ta_features
from user_data.litmus import external_informative_data as eid

import numpy as np  # noqa
import pandas as pd  # noqa

# Prevent pandas complaining with future warning errors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class CointegrationExplore(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "5m"

    # Required. But effectively disabled.
    minimal_roi = {
        # Effectively disabled. Not used for MetaModel. PT & SL levels defined in custom_sell()
        '1000': 1000
    }

    # Stop loss disabled. Use custom_exit() for TBM meta labeling.
    stoploss = -1

    # Turn on position adjustment
    position_adjustment_enable = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count = 100

    def bot_loop_start(self, **kwargs) -> None:

        # Informative: ETH
        self.dataframe_eth_5m = self.dp.get_pair_dataframe(pair='ETH/USDT', timeframe='5m')
        self.dataframe_eth_5m = eid.add_ta_informative(self.dataframe_eth_5m, suffix='_eth')

        # Informative: SOL
        self.dataframe_sol_5m = self.dp.get_pair_dataframe(pair='SOL/USDT', timeframe='5m')
        self.dataframe_sol_5m = eid.add_ta_informative(self.dataframe_sol_5m, suffix='_sol')

        # Informative: 1INCH
        self.dataframe_1inch_5m = self.dp.get_pair_dataframe(pair='1INCH/USDT', timeframe='5m')
        self.dataframe_1inch_5m = eid.add_ta_informative(self.dataframe_1inch_5m, suffix='_1inch')

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign timeframe to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        # Add additional informative pairs
        informative_pairs += [
            ("ETH/USDT", "5m"),
            ("SOL/USDT", "5m"),
            ("1INCH/USDT", "5m")
        ]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:

        # Apply TA indicators to the pair dataframe
        dataframe = add_all_ta_features(
            dataframe, open="open", high="high", low="low",
            close="close", volume="volume",  fillna=True)

        # Merge informative pairs
        # Note: `.copy` needed to avoid `date` column being removed during merge which means column
        # not available for joining for subsequent pairs
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_eth_5m.copy(), self.timeframe,
            '5m', ffill=True, date_column='date_eth')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_sol_5m.copy(), self.timeframe,
            '5m', ffill=True, date_column='date_sol')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_1inch_5m.copy(), self.timeframe,
            '5m', ffill=True, date_column='date_1inch')

        # Add reference to pair so ML model can generate feature for prediction
        dataframe['pair_copy'] = metadata['pair']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy trend indicator when buy_trigger crosses from 0 to 1
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with buy column
        """

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Disable populate exit
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with sell column
        """

        return dataframe
