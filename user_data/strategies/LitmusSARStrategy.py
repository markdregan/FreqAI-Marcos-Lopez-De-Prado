import logging
import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta

from LitmusSimpleStrategy import LitmusSimpleStrategy

from datetime import datetime
from feature_engine.creation import CyclicalFeatures
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from functools import reduce
from freqtrade.litmus.label_helpers import tripple_barrier
from freqtrade.litmus import indicator_helpers as ih
from pandas import DataFrame
from technical import qtpylib
from typing import Optional

logger = logging.getLogger(__name__)


class LitmusSARStrategy(LitmusSimpleStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusSARStrategy
      --config user_data/strategies/config.LitmusMLDP.json
      --freqaimodel LitmusMLDPClassifier --verbose
    """

    # ROI table:
    minimal_roi = {
        "0": 1.0,
        "1000": 0
    }

    # Stoploss:
    stoploss = -0.05

    # Stop loss config
    use_custom_stoploss = False
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.00
    trailing_only_offset_is_reached = False

    # DCA Config
    position_adjustment_enable = False
    max_entry_position_adjustment = 3

    # Other strategy flags
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    startup_candle_count = 120

    plot_config = {
        "main_plot": {
            "ema_200": {"color": "grey"},
        },
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
                "DI_values": {"color": "grey"}
            },
            "SAR": {
                "sar": {"color": "Yellow"}
            },
            "Meta": {
                "a_win_long": {"color": "PaleGreen"},
                "a_win_short": {"color": "Salmon"},
                "meta_enter_long_threshold": {"color": "ForestGreen"},
                "meta_enter_short_threshold": {"color": "FireBrick"},
            },
            "GT": {
                "primary_enter_long_tbm": {"color": "PaleGreen"},
                "primary_enter_short_tbm": {"color": "Salmon"},
            },
            "EE": {
                "primary_enter_long": {"color": "PaleGreen"},
                "primary_enter_short": {"color": "Salmon"},
            },
            "Returns": {
                "value_meta_long_max_returns_&-meta_target": {"color": "PaleGreen"},
                "value_meta_short_max_returns_&-meta_target": {"color": "Salmon"}
            },
            "Time": {
                "total_time_&-meta_target": {"color": "SkyBlue"},
            },
            "Feat": {
                "num_features_selected_&-meta_target": {"color": "SkyBlue"}
            }
        },
    }

    def feature_engineering_standard(self, dataframe, **kwargs) -> pd.DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param df: strategy dataframe which will receive the features
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """

        # Rules Based Entry & Exit Indicators
        dataframe['sar'] = ta.SAR(dataframe["high"], dataframe["low"], acceleration=0.01, maximum=0.2)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=100, price='close')

        dataframe["primary_enter_long"] = (
            (qtpylib.crossed_above(dataframe['sar'], dataframe['sar'].shift(1))) &
            (dataframe['close'] > dataframe['ema_200'])
        )

        dataframe["primary_enter_short"] = (
            (qtpylib.crossed_below(dataframe['sar'], dataframe['sar'].shift(1))) &
            (dataframe['close'] < dataframe['ema_200'])
        )

        dataframe["primary_exit_long"] = (
            qtpylib.crossed_below(dataframe['sar'], dataframe['sar'].shift(1))
        )

        dataframe["primary_exit_short"] = (
            qtpylib.crossed_below(dataframe['sar'], dataframe['sar'].shift(1))
        )

        # Generate remaining features from super class
        dataframe = super().feature_engineering_standard(dataframe, **kwargs)

        return dataframe

