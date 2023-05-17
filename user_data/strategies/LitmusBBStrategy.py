import logging
import pandas as pd
import talib.abstract as ta

from LitmusSimpleStrategy import LitmusSimpleStrategy
from technical import qtpylib

logger = logging.getLogger(__name__)


class LitmusBBStrategy(LitmusSimpleStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusBBStrategy
      --config user_data/strategies/config.LitmusMLDP.json
      --freqaimodel LitmusMLDPClassifier --verbose
    """

    plot_config = {
        "main_plot": {
            "mm_bb_lowerband": {"color": "grey"},
            "mm_bb_upperband": {"color": "grey"},
        },
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
                "DI_values": {"color": "grey"}
            },
            "Meta": {
                "a_win_long": {"color": "PaleGreen"},
                "a_win_short": {"color": "Salmon"},
                "meta_enter_long_threshold": {"color": "ForestGreen"},
                "meta_enter_short_threshold": {"color": "FireBrick"},
            },
            "BB": {
                "rsi": {"color": "Purple"},
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

    # ROI table:
    minimal_roi = {
        "0": 1.0,
        "1000": 0
    }

    # Stoploss:
    stoploss = -0.05

    # Stop loss config
    use_custom_stoploss = True
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
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=25, stds=1.9
        )
        dataframe["mm_bb_lowerband"] = bollinger["lower"]
        dataframe["mm_bb_middleband"] = bollinger["mid"]
        dataframe["mm_bb_upperband"] = bollinger["upper"]

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe["primary_enter_long"] = (
            (dataframe["low"] < dataframe["mm_bb_lowerband"]) &
            (qtpylib.crossed_above(dataframe['rsi'], dataframe['rsi'].shift(1)))
        )
        dataframe["primary_enter_short"] = (
            (dataframe["high"] > dataframe["mm_bb_upperband"]) &
            (qtpylib.crossed_below(dataframe['rsi'], dataframe['rsi'].shift(1)))
        )

        dataframe["primary_exit_long"] = qtpylib.crossed_above(
            dataframe["close"], dataframe["mm_bb_upperband"])
        dataframe["primary_exit_short"] = qtpylib.crossed_below(
            dataframe["close"], dataframe["mm_bb_lowerband"])

        # Generate remaining features from super class
        dataframe = super().feature_engineering_standard(dataframe, **kwargs)

        return dataframe

