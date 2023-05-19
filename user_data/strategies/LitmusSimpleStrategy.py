import logging
import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta

from datetime import datetime, timedelta
from feature_engine.creation import CyclicalFeatures
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from functools import reduce
from freqtrade.litmus.label_helpers import tripple_barrier
from freqtrade.litmus import indicator_helpers as ih
from pandas import DataFrame
from technical import qtpylib
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Config for visual display pandas (for debugging)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)


class LitmusSimpleStrategy(IStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusSimpleStrategy
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
            "Test": {
                "b_win": {"color": "Pink"},
                "c_win": {"color": "Yellow"},
            },
            "GT": {
                "primary_enter_long_tbm": {"color": "PaleGreen"},
                "primary_enter_short_tbm": {"color": "Salmon"},
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

    """prot_cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=True)
    prot_stoploss_enabled = BooleanParameter(default=False, space="protection", optimize=False)
    prot_stoploss_duration = IntParameter(1, 60, default=17, space="protection", optimize=False)"""

    # ROI table:
    minimal_roi = {
        "0": 1.0,
        "1000": 0
    }

    # Stoploss:
    stoploss = -0.02

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

    """@property
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

        return prot"""

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

    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> pd.DataFrame:
        """
        Will expand:
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`

        Will not expand: `indicator_periods_candles`
        """
        t = 20

        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-price_change"] = dataframe["close"].pct_change()
        dataframe["%-volatility_price"] = \
            dataframe["%-price_change"].rolling(t).std() * np.sqrt(t)

        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-volume_change"] = np.log(dataframe['volume'] / dataframe['volume'].shift())
        dataframe["%-volatility_volume"] = \
            dataframe["%-volume_change"].rolling(t).std() * np.sqrt(t)

        return dataframe

    def feature_engineering_standard(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> pd.DataFrame:
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

        # Time features
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        cyclical_transform = CyclicalFeatures(
            variables=["%-day_of_week", "%-hour_of_day"], max_values=None, drop_original=True
        )
        dataframe = cyclical_transform.fit_transform(dataframe)

        # Secondary Model: TBM target features
        secondary_target_params = self.freqai_info["secondary_target_parameters"]

        # Long: TBM Labeling
        window = secondary_target_params["tmb_long_window"]
        params = {
            "upper_pct": secondary_target_params["tmb_long_upper"],
            "lower_pct": secondary_target_params["tmb_long_lower"]
        }
        dataframe["primary_enter_long_tbm"] = (
            dataframe["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )

        # Short: TBM Labeling
        window = secondary_target_params["tmb_short_window"]
        params = {
            "upper_pct": secondary_target_params["tmb_short_upper"],
            "lower_pct": secondary_target_params["tmb_short_lower"]
        }
        dataframe["primary_enter_short_tbm"] = (
            dataframe["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )

        # Crude forward-looking return of current candle (Note: cannot be used as feature)
        dataframe["!-trade_return"] = dataframe["close"].pct_change(window).shift(-window)

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> pd.DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param df: strategy dataframe which will receive the targets
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """

        # Secondary: Meta Model targets
        long_tbm_map = {1: "a_win_long", 0: "loss", -1: "loss"}
        dataframe["long_outcome_tbm"] = dataframe["primary_enter_long_tbm"].map(long_tbm_map)
        short_tbm_map = {1: "loss", 0: "loss", -1: "a_win_short"}
        dataframe["short_outcome_tbm"] = dataframe["primary_enter_short_tbm"].map(short_tbm_map)

        # Merge long/short targets into single column
        conditions = [dataframe["primary_enter_long"], dataframe["primary_enter_short"]]
        choices = [dataframe["long_outcome_tbm"], dataframe["short_outcome_tbm"]]
        dataframe["&-meta_target"] = np.select(conditions, choices, default="drop-row")
        dataframe["&-meta_target"] = dataframe["&-meta_target"].fillna(value="drop-row")

        print(dataframe.groupby("&-meta_target").size())

        # Experiment: Binary classifier
        long_tbm_map = {1: "b_win", 0: "b_loss", -1: "b_loss"}
        dataframe["b_long_outcome_tbm"] = dataframe["primary_enter_long_tbm"].map(long_tbm_map)
        short_tbm_map = {1: "b_loss", 0: "b_loss", -1: "b_win"}
        dataframe["b_short_outcome_tbm"] = dataframe["primary_enter_short_tbm"].map(short_tbm_map)

        conditions = [dataframe["primary_enter_long"], dataframe["primary_enter_short"]]
        choices = [dataframe["b_long_outcome_tbm"], dataframe["b_short_outcome_tbm"]]
        dataframe["&-meta_target_bin"] = np.select(conditions, choices, default="drop-row")
        dataframe["&-meta_target_bin"] = dataframe["&-meta_target_bin"].fillna(value="drop-row")

        # Experiment: Binary classifier per side (just long to test)
        long_tbm_map = {1: "c_win", 0: "c_loss", -1: "c_loss"}
        dataframe["c_long_outcome_tbm"] = dataframe["primary_enter_long_tbm"].map(long_tbm_map)

        conditions = [dataframe["primary_enter_long"]]
        choices = [dataframe["c_long_outcome_tbm"]]
        dataframe["&-meta_target_bin2"] = np.select(conditions, choices, default="drop-row")
        dataframe["&-meta_target_bin2"] = dataframe["&-meta_target_bin2"].fillna(value="drop-row")

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        # Meta: Trigger thresholds
        smoothing_window = self.freqai_info["entry_parameters"].get("smoothing_window", 30)
        dataframe["meta_enter_long_threshold"] = dataframe[
            "threshold_meta_long_max_returns_&-meta_target"].rolling(smoothing_window).mean()
        dataframe["meta_enter_short_threshold"] = dataframe[
            "threshold_meta_short_max_returns_&-meta_target"].rolling(smoothing_window).mean()

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> pd.DataFrame:

        # Long Entry
        conditions = [df["primary_enter_long"],
                      df["a_win_long"] >= df["meta_enter_long_threshold"]]

        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "primary_enter_long")

        # Short Entry
        conditions = [df["primary_enter_short"],
                      df["a_win_short"] >= df["meta_enter_short_threshold"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "primary_enter_short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> pd.DataFrame:

        exit_params = self.freqai_info["exit_parameters"]
        if exit_params.get("exit_trigger_enabled", False):
            # Long Exit
            conditions = [df["primary_exit_long"]]

            if conditions:
                df.loc[
                    reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]
                ] = (1, "primary_exit_long")

            # Short Exit
            conditions = [df["primary_exit_short"]]
            if conditions:
                df.loc[
                    reduce(lambda x, y: x & y, conditions), ["exit_short", "exit_tag"]
                ] = (1, "primary_exit_short")

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
         :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """

        fixed_leverage = self.freqai_info.get("fixed_leverage", 0)
        if fixed_leverage > 0:
            return fixed_leverage
        else:
            return 1.0

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Add tight SL when exit trigger observed
        exit_params = self.freqai_info["exit_parameters"]
        if exit_params.get("sl_exit_trigger_enabled", False):
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()

            if (
                    (last_candle["primary_exit_long"] is True) or
                    (last_candle["primary_exit_short"] is True)):
                logger.info(f"Tightening stoploss as exit trigger detected for {pair}")
                return exit_params.get("sl_exit_trigger_pct", False)

        if current_profit <= 0.00:
            return -1

        if current_profit > 0.01:
            desired_stoploss = current_profit / 2.0

            min_sl = 0.01
            max_sl = 0.03

            new_sl = max(min(desired_stoploss, max_sl), min_sl)

            return new_sl

        return -1

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra buy or sell orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
        """

        # Only allow trade adjustment once every N minutes
        if ((current_time - timedelta(minutes=5) > trade.date_last_filled_utc) &
                (trade.nr_of_successful_entries <= self.max_entry_position_adjustment)):

            df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            last_candle = df.iloc[-1].squeeze()

            # Long Entry
            enter_long = np.where(
                (last_candle["primary_enter_long"] &
                 (last_candle["a_win_long"] >= last_candle["meta_enter_long_threshold"])),
                True, False)

            # Short Entry
            enter_short = np.where(
                (last_candle["primary_enter_short"] &
                 (last_candle["a_win_short"] >= last_candle["meta_enter_short_threshold"])),
                True, False)

            if enter_long or enter_short:
                # This returns first order stake size
                filled_entries = trade.select_filled_orders(trade.entry_side)
                stake_amount = filled_entries[0].cost
                logger.info(f"Trade adjustment made adding {stake_amount} to {trade.pair}")
                return stake_amount

        return None
