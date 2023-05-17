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


class LitmusScalpStrategy(LitmusSimpleStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusScalpStrategy
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
            "ema_high": {"color": "grey"},
            "ema_low": {"color": "grey"},
        },
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
                "DI_values": {"color": "grey"}
            },
            "Scalp": {
                "adx": {"color": "Yellow"},
                "fastk": {"color": "Blue"},
                "fastd": {"color": "Purple"}
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
        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        dataframe["primary_enter_long"] = (
                (dataframe['open'] < dataframe['ema_low']) &
                (dataframe['adx'] > 30) &
                (
                        (dataframe['fastk'] < 30) &
                        (dataframe['fastd'] < 30) &
                        (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
                )
        )

        dataframe["primary_enter_short"] = (
                (dataframe['open'] > dataframe['ema_high']) &
                (dataframe['adx'] > 30) &
                (
                        (dataframe['fastk'] > 70) &
                        (dataframe['fastd'] > 70) &
                        (qtpylib.crossed_below(dataframe['fastk'], dataframe['fastd']))
                )
        )

        dataframe["primary_exit_long"] = (
                (dataframe['open'] >= dataframe['ema_high']) |
                (
                        (qtpylib.crossed_above(dataframe['fastk'], 70)) |
                        (qtpylib.crossed_above(dataframe['fastd'], 70))
                )
        )

        dataframe["primary_exit_short"] = (
                (dataframe['open'] <= dataframe['ema_low']) |
                (
                        (qtpylib.crossed_below(dataframe['fastk'], 30)) |
                        (qtpylib.crossed_below(dataframe['fastd'], 30))
                )
        )

        # Generate remaining features from super class
        dataframe = super().feature_engineering_standard(dataframe, **kwargs)

        return dataframe

