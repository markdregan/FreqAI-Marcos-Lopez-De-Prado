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
                "win_long": {"color": "ForestGreen"},
                "win_short": {"color": "FireBrick"},
                "win_long_enter_threshold": {"color": "Green"},
                "win_short_enter_threshold": {"color": "Red"},
            },
            "RSI": {
                "rsi": {"color": "Purple"},
            },
            "Ent": {
                "primary_enter_long": {"color": "PaleGreen"},
                "primary_enter_short": {"color": "FireBrick"},
            },
            "GT": {
                "primary_enter_long_tbm": {"color": "PaleGreen"},
                "primary_enter_short_tbm": {"color": "FireBrick"},
            },
            "Returns": {
                "win_long_value_&-meta_target_binary_long": {"color": "PaleGreen"},
                "win_short_value_&-meta_target_binary_short": {"color": "FireBrick"},
            },
            "F1": {
                "value_meta_f1_score_&-meta_target_binary_long": {"color": "PaleGreen"},
                "value_meta_f1_score_&-meta_target_binary_short": {"color": "FireBrick"},
            },
            "Feat": {
                "num_features_selected_&-meta_target_binary_long": {"color": "PaleGreen"},
                "num_features_selected_&-meta_target_binary_short": {"color": "FireBrick"}
            },
            "RMax": {
                "!-trade_return_median_max_price_change": {"color": "PaleGreen"},
                "!-trade_return_median_min_price_change": {"color": "FireBrick"}
            }
        },
    }

    # ROI table:
    minimal_roi = {
        "0": 1,
        "1000": 0
    }

    # Stoploss (initial before custom_stoploss kicks in)
    stoploss = -0.05

    # Stop loss config
    use_custom_stoploss = True
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.00
    trailing_only_offset_is_reached = False

    # DCA Config
    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    # Other strategy flags
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    startup_candle_count = 300

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

        # Primary Entry Logic (shift entry forward +1 step)
        dataframe["primary_enter_long"] = (
            (dataframe["low"] < dataframe["mm_bb_lowerband"]) &
            (qtpylib.crossed_above(dataframe["rsi"], dataframe["rsi"].shift(1)))
        )
        dataframe["primary_enter_long"] = dataframe["primary_enter_long"].shift(1).fillna(
            value=False)

        dataframe["primary_enter_short"] = (
            (dataframe["high"] > dataframe["mm_bb_upperband"]) &
            (qtpylib.crossed_below(dataframe["rsi"], dataframe["rsi"].shift(1)))
        )
        dataframe["primary_enter_short"] = dataframe["primary_enter_short"].shift(1).fillna(
            value=False)

        # Primary Exit Logic
        dataframe["primary_exit_long"] = (
                (dataframe["high"] > dataframe["mm_bb_upperband"]) &
                (qtpylib.crossed_below(dataframe['rsi'], dataframe['rsi'].shift(1)))
        )
        dataframe["primary_exit_short"] = (
                (dataframe["low"] < dataframe["mm_bb_lowerband"]) &
                (qtpylib.crossed_above(dataframe["rsi"], dataframe["rsi"].shift(1)))
        )

        # Generate remaining features from super class
        dataframe = super().feature_engineering_standard(dataframe, **kwargs)

        return dataframe
