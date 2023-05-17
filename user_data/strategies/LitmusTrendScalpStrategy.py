import logging
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
            "sma-50": {"color": "Grey"},
            "sma-25": {"color": "Grey"},
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
            "Bin": {
                "b_win_long": {"color": "PaleGreen"},
                "b_win_short": {"color": "Salmon"},
                "meta_bin_enter_long_threshold": {"color": "ForestGreen"},
                "meta_bin_enter_short_threshold": {"color": "FireBrick"},
            },
            "Logic": {
                "rsi-20": {"color": "Purple"},
                "rsi-20-ma": {"color": "Yellow"},
                "rsi-100": {"color": "Grey"}
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
        "0": 0.01,
        "100": 0
    }

    # Stoploss:
    stoploss = -0.05

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
        dataframe["sma-50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["sma-25"] = ta.SMA(dataframe, timeperiod=25)

        dataframe["rsi-20"] = ta.RSI(dataframe, timeperiod=20)
        dataframe["rsi-100"] = ta.RSI(dataframe, timeperiod=100)
        dataframe["rsi-20-ma"] = dataframe["rsi-20"].rolling(window=14).mean()

        dataframe["trend_long"] = (
                (dataframe["sma-25"] > dataframe["sma-50"]) &
                (dataframe["rsi-20"] > dataframe["rsi-100"])
        )
        dataframe["trend_short"] = (
                (dataframe["sma-25"] < dataframe["sma-50"]) &
                (dataframe["rsi-20"] < dataframe["rsi-100"])
        )

        dataframe["primary_enter_long"] = (
            (dataframe["trend_long"]) &
            (qtpylib.crossed_above(dataframe["rsi-20"], dataframe["rsi-20-ma"]))
        )
        dataframe["primary_enter_short"] = (
            (dataframe["trend_short"]) &
            (qtpylib.crossed_below(dataframe["rsi-20"], dataframe["rsi-20-ma"]))
        )

        dataframe["primary_exit_long"] = qtpylib.crossed_below(
            dataframe["rsi-20"], dataframe["rsi-20"].shift(1))
        dataframe["primary_exit_short"] = qtpylib.crossed_above(
            dataframe["rsi-20"], dataframe["rsi-20"].shift(1))

        # Generate remaining features from super class
        dataframe = super().feature_engineering_standard(dataframe, **kwargs)

        return dataframe

    """def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Above 20% profit, sell when rsi < 80
        if current_profit > 0.2:
            if last_candle['rsi'] < 80:
                return 'rsi_below_80'

        # Between 2% and 10%, sell if EMA-long above EMA-short
        if 0.02 < current_profit < 0.1:
            if last_candle['emalong'] > last_candle['emashort']:
                return 'ema_long_below_80'

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 1:
            return 'unclog'"""

