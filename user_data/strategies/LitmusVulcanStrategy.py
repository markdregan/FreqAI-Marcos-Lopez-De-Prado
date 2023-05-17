import logging
import pandas as pd
import talib.abstract as ta

from LitmusSimpleStrategy import LitmusSimpleStrategy

from freqtrade.strategy import BooleanParameter, IntParameter
from technical import qtpylib

logger = logging.getLogger(__name__)


class LitmusVulcanStrategy(LitmusSimpleStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusVulcanStrategy
      --config user_data/strategies/config.LitmusMLDP.json
      --freqaimodel LitmusMLDPClassifier --verbose
    """

    # ROI table:
    minimal_roi = {
        "0": 1.0
    }

    # Stoploss:
    stoploss = -0.25

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
            "Vulcan": {
                "RSI": {"color": "Yellow"},
                "RSI_SMA": {"color": "Blue"},
                "growing_SMA": {"color": "ForestGreen"},
                "shrinking_SMA": {"color": "FireBrick"}
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
        dataframe["RSI"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["RSI_SMA"] = dataframe["RSI"].rolling(window=50).mean()

        dataframe["SMA"] = ta.SMA(dataframe, timeperiod=23)
        dataframe["growing_SMA"] = (
                (dataframe["SMA"] > dataframe["SMA"].shift(1))
                & (dataframe["SMA"].shift(1) > dataframe["SMA"].shift(2))
                & (dataframe["SMA"].shift(2) > dataframe["SMA"].shift(3))
        )
        dataframe["shrinking_SMA"] = (
                (dataframe["SMA"] < dataframe["SMA"].shift(1))
                & (dataframe["SMA"].shift(1) < dataframe["SMA"].shift(2))
                & (dataframe["SMA"].shift(2) < dataframe["SMA"].shift(3))
        )

        stoch = ta.STOCH(
            dataframe,
            fastk_period=14,
            slowk_period=4,
            slowk_matype=0,
            slowd_period=6,
            slowd_matype=0,
        )
        dataframe["slowd"] = stoch["slowd"]
        dataframe["slowk"] = stoch["slowk"]

        dataframe["stoch_long_sell_cross"] = (
            (dataframe["slowd"] > 75) & (dataframe["slowk"] > 75)) & (
            (qtpylib.crossed_below(dataframe["slowk"], dataframe["slowd"]))
        )

        dataframe["stoch_short_sell_cross"] = (
            (dataframe["slowd"] > 75) & (dataframe["slowk"] > 75)) & (
            (qtpylib.crossed_below(dataframe["slowk"], dataframe["slowd"]))
        )

        dataframe["last_lowest"] = dataframe["low"].rolling(100).min().shift(1)
        dataframe["lower_low"] = dataframe["close"] < dataframe["last_lowest"]

        dataframe["last_highest"] = dataframe["high"].rolling(100).max().shift(1)
        dataframe["higher_high"] = dataframe["high"] > dataframe["last_highest"]

        dataframe["primary_enter_long"] = (
                (dataframe["close"] > dataframe["SMA"])
                & (dataframe["growing_SMA"])
                & (dataframe["RSI"] > dataframe["RSI_SMA"])
                & (dataframe["RSI"] > 50)
            )

        dataframe["primary_enter_short"] = (
                (dataframe["close"] < dataframe["SMA"])
                & (dataframe["shrinking_SMA"])
                & (dataframe["RSI"] < dataframe["RSI_SMA"])
                & (dataframe["RSI"] < 50)
            )

        dataframe["primary_exit_long"] = (
                (dataframe["stoch_long_sell_cross"] == True) | (dataframe["lower_low"] == True)
        )

        dataframe["primary_exit_short"] = (
                (dataframe["stoch_short_sell_cross"] == True) | (dataframe["higher_high"] == True)
        )

        # Generate remaining features from super class
        dataframe = super().feature_engineering_standard(dataframe, **kwargs)

        return dataframe

