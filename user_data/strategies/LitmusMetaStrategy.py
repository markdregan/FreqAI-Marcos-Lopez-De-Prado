import numpy as np
from feature_engine.creation import CyclicalFeatures
from freqtrade.strategy import IStrategy, BooleanParameter, IntParameter
from functools import reduce
from freqtrade.litmus.label_helpers import tripple_barrier
from freqtrade.litmus import indicator_helpers as ih
from pandas import DataFrame
from technical import qtpylib

import logging
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta

logger = logging.getLogger(__name__)


class LitmusMetaStrategy(IStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusMetaStrategy
      --config user_data/strategies/config.LitmusMeta.json
      --freqaimodel LitmusMultiTargetClassifier --verbose
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
            "Long": {
                "a_win_long": {"color": "PaleGreen"},
                "fbeta_entry_thresh_&-primary_enter_long": {"color": "Grey"}
            },
            "Short": {
                "a_win_short": {"color": "Salmon"},
                "fbeta_entry_thresh_&-primary_enter_short": {"color": "Grey"},
            },
            "GT": {
                "primary_enter_long_tbm": {"color": "PaleGreen"},
                "primary_enter_short_tbm": {"color": "Salmon"},
            },
            "Trees": {
                "num_trees_&-primary_enter_long": {"color": "PaleGreen"},
                "num_trees_&-primary_enter_short": {"color": "Salmon"}
            },
            "Time": {
                "total_time_&-primary_enter_long": {"color": "PaleGreen"},
                "total_time_&-primary_enter_short": {"color": "Salmon"}
            },
        },
    }

    prot_cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=True)
    prot_stoploss_enabled = BooleanParameter(default=False, space="protection", optimize=False)
    prot_stoploss_duration = IntParameter(1, 60, default=17, space="protection", optimize=False)

    # ROI table:
    minimal_roi = {
        "0": 0.05,
        "80": 0
    }

    # Stoploss:
    stoploss = -0.05

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.198
    trailing_stop_positive_offset = 0.245
    trailing_only_offset_is_reached = False

    # Other strategy flags
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    startup_candle_count = 120

    @property
    def protections(self):
        prot = []
        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.prot_cooldown_lookback.value
        })

        if self.prot_stoploss_enabled.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period": 60,
                "trade_limit": 2,
                "stop_duration_candles": self.prot_stoploss_duration.value,
                "required_profit": 0.0,
                "only_per_pair": False,
                "only_per_side": False
            })

        return prot

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        """
        Will expand:
        `indicator_periods_candles` *`include_timeframes` * `include_shifted_candles`
        * `include_corr_pairs`
        """

        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-cci"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er"] = pta.er(dataframe['close'], length=period)
        dataframe["%-rocr"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-cmf"] = ih.chaikin_mf(dataframe, periods=period)
        dataframe["%-tcp"] = ih.top_percent_change(dataframe, period)
        dataframe["%-cti"] = pta.cti(dataframe['close'], length=period)
        dataframe["%-chop"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear"] = ta.LINEARREG_ANGLE(dataframe['close'], timeperiod=period)

        dataframe["%-obv"] = ta.OBV(dataframe)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        dataframe["%-bb_width"] = (
            dataframe["bb_upperband"]
            - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]
        dataframe["%-close-bb_lower"] = (
            dataframe["close"] / dataframe["bb_lowerband"]
        )

        dataframe["%-roc"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        # Absolute Price Oscillator
        dataframe["%-apo"] = ta.APO(
            dataframe["close"], fastperiod=int(period / 2), slowperiod=period, matype=0)

        # PPO (Percentage Price Oscilator)
        dataframe["%-ppo"] = ta.PPO(
            dataframe["close"], fastperiod=int(period / 2), slowperiod=period, matype=0)

        # MACD (macd, macdsignal, macdhist)
        _, _, macdhist = ta.MACD(
            dataframe["close"], fastperiod=int(period / 2),
            slowperiod=period, signalperiod=int(3 * period / 4))
        dataframe["%-macdhist"] = macdhist

        # Average True Range
        dataframe["%-atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period)

        return dataframe

    def feature_engineering_expand_basic(self, dataframe, **kwargs) -> pd.DataFrame:
        """
        Will expand:
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`

        Will not expand: `indicator_periods_candles`
        """

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

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

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        cyclical_transform = CyclicalFeatures(
            variables=["%-day_of_week", "%-hour_of_day"], max_values=None, drop_original=True
        )
        dataframe = cyclical_transform.fit_transform(dataframe)

        return dataframe

    def set_freqai_targets(self, dataframe, **kwargs) -> pd.DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param df: strategy dataframe which will receive the targets
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """

        # Indicators for primary
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=25, stds=1.9
        )
        dataframe["mm_bb_lowerband"] = bollinger["lower"]
        dataframe["mm_bb_middleband"] = bollinger["mid"]
        dataframe["mm_bb_upperband"] = bollinger["upper"]

        # Primary: Enter Long
        """dataframe["primary_enter_long"] = qtpylib.crossed_above(
            dataframe["close"], dataframe["mm_bb_lowerband"])"""
        dataframe["primary_enter_long"] = np.where(
            dataframe["close"] < dataframe["mm_bb_lowerband"], True, False)

        # Long: TBM Labeling
        logger.info("Starting TBM: Long")
        window = 10
        params = {"upper_pct": 0.01, "lower_pct": 0.01}
        dataframe["primary_enter_long_tbm"] = (
            dataframe["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )
        tbm_map = {1: "a_win_long", 0: "loss_long", -1: "loss_long"}
        dataframe["&-primary_enter_long"] = dataframe["primary_enter_long_tbm"].map(tbm_map)

        # Long: Primary model performance a feature for meta model
        primary_long_perf_df = dataframe.loc[
            dataframe["primary_enter_long"]]["primary_enter_long_tbm"]
        dataframe["%-primary_long_perf_mean"] = primary_long_perf_df.rolling(10).mean()
        dataframe["%-primary_long_perf_std"] = primary_long_perf_df.rolling(10).std()

        # Change target labels when out of scope of primary entries
        dataframe["&-primary_enter_long"] = np.where(
            dataframe["primary_enter_long"], dataframe["&-primary_enter_long"], "drop-row")
        logger.info("Finished TBM: Long")
        logger.info("Label counts for primary long: ")
        print(dataframe.groupby("&-primary_enter_long").size())

        # Primary: Enter Short
        """dataframe["primary_enter_short"] = qtpylib.crossed_below(
            dataframe["close"], dataframe["mm_bb_upperband"])"""
        dataframe["primary_enter_short"] = np.where(
            dataframe["close"] > dataframe["mm_bb_upperband"], True, False)

        # Short: TBM Labeling
        logger.info("Starting TBM: Short")
        window = 10
        params = {"upper_pct": 0.01, "lower_pct": 0.01}
        dataframe["primary_enter_short_tbm"] = (
            dataframe["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )
        tbm_map = {1: "loss_short", 0: "loss_short", -1: "a_win_short"}
        dataframe["&-primary_enter_short"] = dataframe["primary_enter_short_tbm"].map(tbm_map)

        # Short: Primary model performance a feature for meta model
        primary_short_perf_df = dataframe.loc[
            dataframe["primary_enter_short"]]["primary_enter_short_tbm"]
        dataframe["%-primary_short_perf_mean"] = primary_short_perf_df.rolling(10).mean()
        dataframe["%-primary_short_perf_std"] = primary_short_perf_df.rolling(10).std()

        # Change target labels when out of scope of primary entries
        dataframe["&-primary_enter_short"] = np.where(
            dataframe["primary_enter_short"], dataframe["&-primary_enter_short"], "drop-row")
        logger.info("Finished TBM: Short")
        logger.info("Label counts for primary short: ")
        print(dataframe.groupby("&-primary_enter_short").size())

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> pd.DataFrame:

        # TODO: Populate meta model thresholds to be exceeded

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> pd.DataFrame:

        # TODO: Update using predictions from meta model
        # Long Entry
        conditions = [df["primary_enter_long"],
                      df["a_win_long"] >= df["fbeta_entry_thresh_&-primary_enter_long"]]

        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "primary_enter_long")

        # Short Entry
        conditions = [df["primary_enter_short"],
                      df["a_win_short"] >= df["fbeta_entry_thresh_&-primary_enter_short"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "primary_enter_short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> pd.DataFrame:

        # TODO: exit using stop loss and profit taking only for V1

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])
