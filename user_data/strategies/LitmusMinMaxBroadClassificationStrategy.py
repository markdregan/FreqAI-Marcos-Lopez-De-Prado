from datetime import datetime

from feature_engine.creation import CyclicalFeatures
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, BooleanParameter, DecimalParameter, IntParameter
from functools import reduce
from freqtrade.litmus.label_helpers import nearby_extremes
from freqtrade.litmus import indicator_helpers as ih
from pandas import DataFrame
from technical import qtpylib
from typing import Optional

import logging
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
import zigzag

logger = logging.getLogger(__name__)


class LitmusMinMaxBroadClassificationStrategy(IStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusMinMaxBroadClassificationStrategy
      --config user_data/strategies/config.LitmusMinMaxBroadClassification.json
      --freqaimodel LitmusMultiTargetClassifier --verbose
    """

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
                "DI_values": {"color": "grey"}
            },
            "M0": {
                "minima_0": {"color": "LightGrey"},
                "long_entry_target": {"color": "PaleGreen"},
                "short_exit_target": {"color": "Maroon"},
                "maxima_0": {"color": "Grey"},
                "short_entry_target": {"color": "Salmon"},
                "long_exit_target": {"color": "ForestGreen"},
            },
            "M1": {
                "minima_1": {"color": "LightGrey"},
                "short_exit_missed_target": {"color": "Maroon"},
                "maxima_1": {"color": "Grey"},
                "long_exit_missed_target": {"color": "ForestGreen"},
            },
            "F1": {
                "max_fbeta_entry_maxima_0": {"color": "fae243"},
                "max_fbeta_entry_minima_0": {"color": "fae243"},
                "max_fbeta_exit_maxima_1": {"color": "43dcc7"},
                "max_fbeta_exit_minima_1": {"color": "43dcc7"},
            },
            "Labels": {
                "raw_peaks_1": {"color": "#ffffa3"},
                "nearby_peaks_1": {"color": "#e0ce38"},
                "raw_peaks_0": {"color": "#a47ebc"},
                "nearby_peaks_0": {"color": "#700CBC"}
            },
            "Other": {
                "total_time": {"color": "Pink"},
                "num_trees_&target_0": {"color": "Orange"},
                "num_trees_&target_1": {"color": "#65ceff"}
            },
        },
    }

    # Hyperopt parameters
    long_entry_mul = DecimalParameter(0.5, 3, decimals=1, default=0.8, space="buy", optimize=True)
    short_entry_mul = DecimalParameter(0.5, 3, decimals=1, default=0.7, space="buy", optimize=True)
    long_exit_mul = DecimalParameter(0.5, 3, decimals=1, default=20, space="sell", optimize=True)
    short_exit_mul = DecimalParameter(0.5, 3, decimals=1, default=20, space="sell", optimize=True)

    do_predict_enabled = BooleanParameter(default=False, space="protection", optimize=False)
    do_predict_threshold = IntParameter(-2, 1, default=1, space="protection", optimize=False)
    DI_threshold_enabled = BooleanParameter(default=False, space="protection", optimize=False)
    DI_threshold = DecimalParameter(
        0.5, 2, decimals=1, default=2, space="protection", optimize=False)

    prot_cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=True)
    prot_stoploss_enabled = BooleanParameter(default=False, space="protection", optimize=False)
    prot_stoploss_duration = IntParameter(1, 60, default=17, space="protection", optimize=False)

    # Buy hyperspace params:
    buy_params = {
        "long_entry_mul": 0.9,
        "short_entry_mul": 0.6,
    }

    # Sell hyperspace params:
    sell_params = {
        "long_exit_mul": 2.9,
        "short_exit_mul": 2.1,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.112,
        "8": 0.041,
        "39": 0.025,
        "81": 0
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.198
    trailing_stop_positive_offset = 0.245
    trailing_only_offset_is_reached = False

    # Other strategy flags
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    startup_candle_count = 200

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
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param df: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
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
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
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
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param df: strategy dataframe which will receive the features
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
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

        # Zigzag min/max for pivot positions
        for i, g in enumerate(self.freqai_info["labeling_parameters"]["zigzag_min_growth"]):
            logger.info(f"Starting zigzag labeling method ({i})")
            min_growth = self.freqai_info["labeling_parameters"]["zigzag_min_growth"][i]
            peaks = zigzag.peak_valley_pivots(
                dataframe["close"].values, min_growth, -min_growth)

            peaks[0] = 0  # Set first value of peaks = 0
            peaks[-1] = 0  # Set last value of peaks = 0

            name_map = {0: f"not_minmax_{i}", 1: f"maxima_{i}", -1: f"minima_{i}"}

            # Smear label to values nearby within threshold
            dataframe[f"raw_peaks_{i}"] = peaks
            nearby_threshold = self.freqai_info["labeling_parameters"]["nearby_threshold"][i]
            dataframe[f"nearby_peaks_{i}"] = nearby_extremes(
                dataframe[["close", f"raw_peaks_{i}"]],
                threshold=nearby_threshold,
                forward_pass=self.freqai_info["labeling_parameters"]["forward_pass"][i],
                reverse_pass=self.freqai_info["labeling_parameters"]["reverse_pass"][i])
            dataframe[f"&target_{i}"] = dataframe[f"nearby_peaks_{i}"].map(name_map)

            # Shift target for benefit of hindsight predictions
            target_offset = self.freqai_info["labeling_parameters"]["target_offset"][i]
            dataframe[f"nearby_peaks_{i}"] = dataframe[f"nearby_peaks_{i}"].shift(
                target_offset).fillna(value=0)

            dataframe[f"real_peaks_{i}"] = peaks

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        smoothing_win = self.freqai_info["trigger_parameters"].get("smoothing_window", 1)

        # Long entry
        dataframe["long_entry_target"] = (
                dataframe["fbeta_entry_thresh_minima_0"].rolling(smoothing_win).mean()
        )

        # Long exit
        dataframe["long_exit_target"] = (
                dataframe["fbeta_exit_thresh_maxima_0"].rolling(smoothing_win).mean()
        )

        # Long exit missed
        dataframe["long_exit_missed_target"] = (
            dataframe["fbeta_exit_thresh_maxima_1"].rolling(smoothing_win).mean()
        )

        # Short entry
        dataframe["short_entry_target"] = (
                dataframe["fbeta_entry_thresh_maxima_0"].rolling(smoothing_win).mean()
        )

        # Short exit
        dataframe["short_exit_target"] = (
                dataframe["fbeta_exit_thresh_minima_0"].rolling(smoothing_win).mean()
        )

        # Short exit missed
        dataframe["short_exit_missed_target"] = (
            dataframe["fbeta_exit_thresh_minima_1"].rolling(smoothing_win).mean()
        )

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> pd.DataFrame:

        # Long Entry
        conditions = []
        if self.DI_threshold_enabled.value:
            conditions.append(df["DI_values"] < self.DI_threshold.value)
        if self.do_predict_enabled.value:
            conditions.append(df["do_predict"] >= self.do_predict_threshold.value)
        conditions.append(
            qtpylib.crossed_above(
                df["minima_0"], df["long_entry_target"])
        )
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "minima_entry")

        # Short Entry
        conditions = []
        if self.DI_threshold_enabled.value:
            conditions.append(df["DI_values"] < self.DI_threshold.value)
        if self.do_predict_enabled.value:
            conditions.append(df["do_predict"] >= self.do_predict_threshold.value)
        conditions.append(
            qtpylib.crossed_above(
                df["maxima_0"], df["short_entry_target"])
        )
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "maxima_entry")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> pd.DataFrame:

        # Long Exit
        conditions = []
        conditions.append(
            qtpylib.crossed_above(
                df["maxima_0"], df["long_exit_target"])
        )
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]
            ] = (1, "maxima_exit")

        # Long Exit Missed
        conditions = []
        conditions.append(
            qtpylib.crossed_above(
                df["maxima_1"], df["long_exit_missed_target"])
        )
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]
            ] = (1, "maxima_missed_exit")

        # Short Exit
        conditions = []
        conditions.append(
            qtpylib.crossed_above(
                df["minima_0"], df["short_exit_target"])
        )
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_short", "exit_tag"]
            ] = (1, "minima_exit")

        # Short Exit Missed
        conditions = []
        conditions.append(
            qtpylib.crossed_above(
                df["minima_1"], df["short_exit_missed_target"])
        )
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_short", "exit_tag"]
            ] = (1, "minima_missed_exit")

        # DI Outlier Exit
        conditions = []
        if self.DI_threshold_enabled.value:
            conditions.append(df["DI_values"] < self.DI_threshold.value)
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_short", "exit_tag"]
            ] = (1, 1, "DI_outlier_exit")

        # Do Predict Outlier Exit
        conditions = []
        if self.do_predict_enabled.value:
            conditions.append(df["do_predict"] >= self.do_predict_threshold.value)
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_short", "exit_tag"]
            ] = (1, 1, "do_predict_outlier_exit")

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Minmax triggers when in profit only
        if current_profit > 0:
            if last_candle["maxima_0"] > \
                    last_candle["long_exit_target"] * self.long_exit_mul.value:
                return "maxima_exit_profit"
            elif last_candle["minima_0"] > \
                    last_candle["short_exit_target"] * self.short_exit_mul.value:
                return "minima_exit_profit"
                """

        """# Between 2% and 10%, sell if EMA-long above EMA-short
        if 0.02 < current_profit < 0.1:
            if last_candle['emalong'] > last_candle['emashort']:
                return 'ema_long_below_80'

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 1:
            return 'unclog'
            """

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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:

        """open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))

        # Balance longs vs shorts to help protect against black swan event
        max_open_trades = self.config.get("max_open_trades", 0)
        if max_open_trades > 0:
            num_shorts, num_longs = 0, 0
            for trade in open_trades:
                if trade.enter_tag == "short":
                    num_shorts += 1
                elif trade.enter_tag == "long":
                    num_longs += 1

            if side == "long" and num_longs >= max_open_trades / 2.0:
                return False

            if side == "short" and num_shorts >= max_open_trades / 2.0:
                return False"""

        # Prevent taking trades that have already moved too far in predicted direction
        """df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                logger.info(f"Trade entry blocked (long) for {pair}")
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                logger.info(f"Trade entry blocked (short) for {pair}")
                return False"""

        return True

    """use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        # Long Exit
        if last_candle["maxima_1"] > last_candle["long_exit_target"]:
            # Tighten stop loss under latest close
            return 0.02

        # Short Exit
        if last_candle["minima_1"] > last_candle["short_exit_target"]:
            return 0.02

        # Otherwise keep current stoploss
        return -1"""

    """
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: Optional[str], **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        latest = dataframe.iloc[-1].squeeze()

        # Model Performance
        if entry_tag == "minima_entry":  # Long entry
            pass
        elif entry_tag == "maxima_entry":  # Short entry
            pass

        # Pair performance

        # Regime familiarity
        mean_stake = proposed_stake  # self.config['stake_amount']
        if latest['DI_value_mean'] == 0:
            stakesize = mean_stake
        else:
            stakesize = mean_stake * (
                0.1 * (latest['DI_value_mean'] - latest['DI_values']) / latest['DI_value_std'] + 1
            )

        # Combine multipliers

        return stakesize"""
