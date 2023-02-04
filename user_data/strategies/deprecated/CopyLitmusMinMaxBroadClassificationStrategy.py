from datetime import datetime

from feature_engine.creation import CyclicalFeatures
from freqtrade.persistence import Trade
from freqtrade.strategy import (IStrategy, merge_informative_pair,
                                BooleanParameter, DecimalParameter, IntParameter)
from functools import reduce
from freqtrade.litmus.label_helpers import nearby_extremes
from pandas import DataFrame
from technical import qtpylib
from typing import Optional

import logging
import pandas as pd
import talib.abstract as ta
import zigzag

logger = logging.getLogger(__name__)


class CopyLitmusMinMaxBroadClassificationStrategy(IStrategy):
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
                "DI_values": {"color": "grey"},
                "DI_no_outlier_detected": {"color": "#1e5780"},
            },
            "Long": {
                "minima_0": {"color": "PaleGreen"},
                "long_entry_target": {"color": "ForestGreen"},
                "maxima_1": {"color": "Salmon"},
                "long_exit_target": {"color": "Crimson"},
            },
            "Short": {
                "maxima_0": {"color": "PaleGreen"},
                "short_entry_target": {"color": "ForestGreen"},
                "minima_1": {"color": "Salmon"},
                "short_exit_target": {"color": "Crimson"},
            },
            "F1": {
                "max_fbeta_entry_maxima_0_&target_0": {"color": "Salmon"},
                "max_fbeta_entry_minima_0_&target_0": {"color": "PaleGreen"},
                "max_fbeta_exit_maxima_1_&target_1": {"color": "Crimson"},
                "max_fbeta_exit_minima_1_&target_1": {"color": "ForestGreen"},
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
    stoploss = -0.269

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.198
    trailing_stop_positive_offset = 0.245
    trailing_only_offset_is_reached = False

    # Other strategy flags
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True

    def informative_pairs(self):
        whitelist_pairs = self.dp.current_whitelist()
        corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        informative_pairs = []
        for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue  # avoid duplication
                informative_pairs.append((pair, tf))
        return informative_pairs

    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):
        """
        Function designed to automatically generate, name and merge features
        from user indicated timeframes in the configuration file. User controls the indicators
        passed to the training/prediction by prepending indicators with `'%-' + coin `
        (see convention below). I.e. user should not prepend any supporting metrics
        (e.g. bb_lowerband below) with % unless they explicitly want to pass that metric to the
        model.
        :param pair: pair to be used as informative
        :param df: strategy dataframe which will receive merges from informatives
        :param tf: timeframe of the dataframe which will modify the feature names
        :param informative: the dataframe associated with the informative pair
        """

        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            informative[f"%%-{coin}-rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%%-{coin}-mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%%-{coin}-adx-period_{t}"] = ta.ADX(informative, window=t)
            informative[f"{coin}-sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            informative[f"{coin}-ema-period_{t}"] = ta.EMA(informative, timeperiod=t)
            informative[f"%-{coin}-close_over_sma-period_{t}"] = (
                informative["close"] / informative[f"{coin}-sma-period_{t}"]
            )

            informative[f"%-{coin}-mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)

            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(informative), window=t, stds=2.2
            )
            informative[f"{coin}-bb_lowerband-period_{t}"] = bollinger["lower"]
            informative[f"{coin}-bb_middleband-period_{t}"] = bollinger["mid"]
            informative[f"{coin}-bb_upperband-period_{t}"] = bollinger["upper"]

            informative[f"%-{coin}-bb_width-period_{t}"] = (
                informative[f"{coin}-bb_upperband-period_{t}"]
                - informative[f"{coin}-bb_lowerband-period_{t}"]
            ) / informative[f"{coin}-bb_middleband-period_{t}"]
            informative[f"%-{coin}-close-bb_lower-period_{t}"] = (
                informative["close"] / informative[f"{coin}-bb_lowerband-period_{t}"]
            )

            informative[f"%-{coin}-roc-period_{t}"] = ta.ROC(informative, timeperiod=t)

            informative[f"%-{coin}-relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            )

            # Absolute Price Oscillator
            informative[f"%-{coin}-apo-period_{t}"] = ta.APO(
                informative["close"], fastperiod=int(t / 2), slowperiod=t, matype=0)

            # PPO (Percentage Price Oscilator)
            informative[f"%-{coin}-ppo-period_{t}"] = ta.PPO(
                informative["close"], fastperiod=int(t / 2), slowperiod=t, matype=0)

            # MACD (macd, macdsignal, macdhist)
            _, _, macdhist = ta.MACD(
                informative["close"], fastperiod=int(t / 2),
                slowperiod=t, signalperiod=int(3 * t / 4))
            informative[f"%-{coin}-macdhist-period_{t}"] = macdhist

            # Average True Range
            informative[f"%-{coin}-atr-period_{t}"] = ta.ATR(
                informative["high"], informative["low"], informative["close"], timeperiod=t)

        informative[f"%-{coin}-pct-change"] = informative["close"].pct_change()
        informative[f"%-{coin}-raw_volume"] = informative["volume"]
        informative[f"%-{coin}-raw_price"] = informative["close"]

        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # Add generalized indicators here (because in live, it will call this
        # function to populate indicators during training). Notice how we ensure not to
        # add them multiple times
        if set_generalized_indicators:
            df["day_of_week"] = df["date"].dt.dayofweek
            df["hour_of_day"] = df["date"].dt.hour
            cyclical_transform = CyclicalFeatures(
                variables=["day_of_week", "hour_of_day"], max_values=None, drop_original=True
            )
            df = cyclical_transform.fit_transform(df)

            # Zigzag min/max for pivot positions
            for i, g in enumerate(self.freqai_info["labeling_parameters"]["zigzag_min_growth"]):
                logger.info(f"Starting zigzag labeling method ({i})")
                min_growth = self.freqai_info["labeling_parameters"]["zigzag_min_growth"][i]
                peaks = zigzag.peak_valley_pivots(
                    df["close"].values, min_growth, -min_growth)

                peaks[0] = 0  # Set first value of peaks = 0
                peaks[-1] = 0  # Set last value of peaks = 0

                name_map = {0: f"not_minmax_{i}", 1: f"maxima_{i}", -1: f"minima_{i}"}

                # Shift target for benefit of hindsight predictions
                target_offset = self.freqai_info["labeling_parameters"]["target_offset"][i]
                df[f"raw_peaks_{i}"] = peaks
                df[f"raw_peaks_{i}"] = df[f"raw_peaks_{i}"].shift(target_offset).fillna(value=0)

                # Smear label to values nearby within threshold
                nearby_threshold = self.freqai_info["labeling_parameters"]["nearby_threshold"][i]
                df[f"nearby_peaks_{i}"] = nearby_extremes(
                    df[["close", f"raw_peaks_{i}"]],
                    threshold=nearby_threshold,
                    forward_pass=self.freqai_info["labeling_parameters"]["forward_pass"][i],
                    reverse_pass=self.freqai_info["labeling_parameters"]["reverse_pass"][i])
                df[f"&target_{i}"] = df[f"nearby_peaks_{i}"].map(name_map)

                df[f"real_peaks_{i}"] = peaks

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)

        smoothing_win = self.freqai_info["trigger_parameters"].get("smoothing_window", 1)

        # Long entry
        dataframe["long_entry_target"] = (
                dataframe["fbeta_entry_thresh_minima_0_&target_0"].rolling(smoothing_win).mean()
        )

        # Long exit
        dataframe["long_exit_target"] = (
                dataframe["fbeta_exit_thresh_maxima_0_&target_0"].rolling(smoothing_win).mean()
        )

        # Short entry
        dataframe["short_entry_target"] = (
                dataframe["fbeta_entry_thresh_maxima_0_&target_0"].rolling(smoothing_win).mean()
        )

        # Short exit
        dataframe["short_exit_target"] = (
                dataframe["fbeta_exit_thresh_minima_0_&target_0"].rolling(smoothing_win).mean()
        )

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Long Entry
        conditions = []
        if self.DI_threshold_enabled.value:
            conditions.append(df["DI_values"] < self.DI_threshold.value)
        """if self.do_predict_enabled.value:
            conditions.append(df["do_predict"] >= self.do_predict_threshold.value)"""
        conditions.append(
            qtpylib.crossed_above(
                df["minima_0"], df["long_entry_target"] * self.long_entry_mul.value)
        )
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "minima_entry")

        # Short Entry
        conditions = []
        if self.DI_threshold_enabled.value:
            conditions.append(df["DI_values"] < self.DI_threshold.value)
        """if self.do_predict_enabled.value:
            conditions.append(df["do_predict"] >= self.do_predict_threshold.value)"""
        conditions.append(
            qtpylib.crossed_above(
                df["maxima_0"], df["short_entry_target"] * self.short_entry_mul.value)
        )
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "maxima_entry")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

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

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Minmax triggers when in profit only
        if current_profit > 0:
            if last_candle["maxima_0"] > \
                    last_candle["long_exit_target"] * self.long_exit_mul.value:
                return "maxima_exit_profit"
            elif last_candle["minima_0"] > \
                    last_candle["short_exit_target"] * self.short_exit_mul.value:
                return "minima_exit_profit"

        """# Between 2% and 10%, sell if EMA-long above EMA-short
        if 0.02 < current_profit < 0.1:
            if last_candle['emalong'] > last_candle['emashort']:
                return 'ema_long_below_80'

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 1:
            return 'unclog'"""

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
