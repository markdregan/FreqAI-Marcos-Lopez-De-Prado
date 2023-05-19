import logging

import numpy as np
import pandas as pd
import talib.abstract as ta

from LitmusSimpleStrategy import LitmusSimpleStrategy
from technical import qtpylib

logger = logging.getLogger(__name__)


class LitmusTrendScalpStrategy(LitmusSimpleStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusTrendScalpStrategy
      --config user_data/strategies/config.LitmusMLDP.json
      --freqaimodel LitmusMLDPClassifier --verbose
    """

    plot_config = {
        "main_plot": {
            "sma-200": {"color": "LightPink"},
            "sma-50": {"color": "LightSalmon"},
            "sma-21": {"color": "LightCyan"},
        },
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "Brown"},
                "DI_values": {"color": "Grey"}
            },
            "Meta": {
                "a_win_long": {"color": "PaleGreen"},
                "a_win_short": {"color": "Salmon"},
                "meta_enter_long_threshold": {"color": "ForestGreen"},
                "meta_enter_short_threshold": {"color": "FireBrick"},
            },
            "RSI": {
                "rsi-14-ma-50": {"color": "Purple"},
                "rsi-14": {"color": "Yellow"}
            },
            "GT": {
                "primary_enter_long_tbm": {"color": "PaleGreen"},
                "primary_enter_short_tbm": {"color": "Salmon"},
            },
            "Entry": {
                "primary_enter_long": {"color": "PaleGreen"},
                "primary_enter_short": {"color": "Salmon"},
                "trend_long": {"color": "Green"},
                "trend_short": {"color": "Red"}
            },
            "Exit": {
                "primary_exit_long": {"color": "Green"},
                "primary_exit_short": {"color": "Red"}
            },
            "Returns": {
                "value_meta_long_max_returns_&-meta_target": {"color": "PaleGreen"},
                "value_meta_short_max_returns_&-meta_target": {"color": "Salmon"}
            },
            "Diag": {
                "total_time_&-meta_target": {"color": "Pink"},
                "num_features_selected_&-meta_target": {"color": "Orange"}
            },
            "F1": {
                "value_meta_f1_score_&-meta_target": {"color": "Yellow"}
            }
        },
    }

    # ROI table:
    minimal_roi = {
        "0": 0.02,
        "100": 0
    }

    # Stoploss:
    stoploss = -0.05

    # Other strategy flags
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True

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
        dataframe["sma-200"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["sma-50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["sma-21"] = ta.SMA(dataframe, timeperiod=21)

        dataframe["trend_long"] = (
                # (dataframe["sma-21"] > dataframe["sma-21"].shift(1)) &
                (dataframe["sma-21"] > dataframe["sma-50"]) &
                (dataframe["sma-50"] > dataframe["sma-200"])
        )

        dataframe["trend_short"] = (
                # (dataframe["sma-21"] < dataframe["sma-21"].shift(1)) &
                (dataframe["sma-21"] < dataframe["sma-50"]) &
                (dataframe["sma-50"] < dataframe["sma-200"])
        )

        dataframe["!-trend_long"] = np.where(dataframe["trend_long"], 1, 0)
        dataframe["!-trend_short"] = np.where(dataframe["trend_short"], 1, 0)

        dataframe["rsi-14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi-14-ma-50"] = dataframe["rsi-14"].rolling(window=50).mean()

        dataframe["primary_enter_long"] = (
            (dataframe["trend_long"]) &
            (qtpylib.crossed_above(dataframe["rsi-14"], dataframe["rsi-14-ma-50"]))
        )
        dataframe["primary_enter_short"] = (
            (dataframe["trend_short"]) &
            (qtpylib.crossed_below(dataframe["rsi-14"], dataframe["rsi-14-ma-50"]))
        )

        """print(dataframe.groupby("trend_long").size())
        print(dataframe.groupby("trend_short").size())

        print(dataframe.groupby("primary_enter_long").size())
        print(dataframe.groupby("primary_enter_short").size())"""

        dataframe["primary_exit_long"] = np.where(
                (dataframe["sma-21"] > dataframe["sma-50"]) &
                (dataframe["sma-50"] > dataframe["sma-200"]), False, True
        )

        dataframe["primary_exit_short"] = np.where(
                (dataframe["sma-21"] < dataframe["sma-50"]) &
                (dataframe["sma-50"] < dataframe["sma-200"]), False, True
        )

        # Generate remaining features from super class
        dataframe = super().feature_engineering_standard(dataframe, **kwargs)

        return dataframe
