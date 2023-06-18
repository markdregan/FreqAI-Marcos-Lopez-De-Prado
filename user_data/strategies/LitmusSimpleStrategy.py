import logging
import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta

from cointanalysis import CointAnalysis
from datetime import datetime, timedelta
from feature_engine.creation import CyclicalFeatures
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from functools import reduce
from freqtrade.litmus import bet_sizing
from freqtrade.litmus.label_helpers import tripple_barrier
from freqtrade.litmus import indicator_helpers as ih
from freqtrade.litmus import nilux_indicator_helpers as nih
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
    stoploss = -0.05

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

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):
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

        # Hurst Exp
        dataframe["%-hurst-exp"] = nih.hurst_exponent(dataframe, lookback=200)

        # Stochastic Momentum Index
        dataframe["%-smi"] = nih.smi_momentum(dataframe[["high", "low", "close"]])
        dataframe["%-smi-direction"] = np.where(
            dataframe["%-smi"] > dataframe["%-smi"].shift(1), 1, 0)
        dataframe["%-smi"] = nih.smi_momentum(dataframe[["high", "low", "close"]])

        # Nilux Indicators
        """dataframe[['fisher_cg', 'fisher_sig']] = nih.fisher_cg(dataframe[['high', 'low']])
        dataframe = nih.exhaustion_bars(dataframe)
        dataframe = nih.pinbar(dataframe, dataframe["smi"])
        dataframe = nih.breakouts(dataframe)"""

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
            variables=["%-day_of_week", "%-hour_of_day"],
            max_values={"%-day_of_week": 7, "%-hour_of_day": 24},
            drop_original=True
        )
        dataframe = cyclical_transform.fit_transform(dataframe)

        informative = self.dp.historic_ohlcv(pair="BTC/USDT:USDT", timeframe="5m")

        # Informative features
        coint_df = merge_informative_pair(
            dataframe[["date", "close"]],
            informative[["date", "close"]],
            self.timeframe, self.timeframe, ffill=True)
        coint_df = coint_df[["close", "close_5m"]]

        # Simple BTC derivative features
        dataframe["%-close_div_btc"] = coint_df["close"] / coint_df["close_5m"]
        dataframe["%-close_div_btc_change"] = dataframe["%-close_div_btc"].pct_change()
        dataframe["%-close_div_btc_change_std"] = \
            dataframe["%-close_div_btc_change"].rolling(50).std() * np.sqrt(50)
        dataframe["%-close_div_btc_ppo"] = ta.PPO(
            dataframe["%-close_div_btc"], fastperiod=10, slowperiod=20, matype=0)

        # Cointegration with BTC features
        coint_params = self.freqai_info["feature_engineering"]["cointegration"]
        if coint_params["enabled"]:
            coint = CointAnalysis()
            coint_pvalue = coint.test(coint_df).pvalue_
            dataframe["%%-coint_spread_btc"] = coint.fit_transform(coint_df).reshape(-1, 1)
            dataframe["coint_pvalue"] = coint_pvalue
            logger.info(f"Cointegration pvalue for {metadata['pair']} is {coint_pvalue}")

        # TBM params
        secondary_target_params = self.freqai_info["secondary_target_parameters"]
        window = secondary_target_params["tmb_long_window"]

        # Secondary Model: TBM target features
        params = {
            "upper_pct": secondary_target_params["tmb_long_upper"],
            "lower_pct": secondary_target_params["tmb_long_lower"],
            "result": "side"
        }
        dataframe["primary_enter_long_tbm"] = (
            dataframe["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )

        params = {
            "upper_pct": secondary_target_params["tmb_long_upper"],
            "lower_pct": secondary_target_params["tmb_long_lower"],
            "result": "value"
        }
        dataframe["primary_enter_long_tbm_value"] = (
            dataframe["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )

        params = {
            "upper_pct": secondary_target_params["tmb_short_upper"],
            "lower_pct": secondary_target_params["tmb_short_lower"],
            "result": "side"
        }
        dataframe["primary_enter_short_tbm"] = (
            dataframe["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )

        params = {
            "upper_pct": secondary_target_params["tmb_short_upper"],
            "lower_pct": secondary_target_params["tmb_short_lower"],
            "result": "value"
        }
        dataframe["primary_enter_short_tbm_value"] = (
            dataframe["close"]
            .shift(-window)
            .rolling(window + 1)
            .apply(tripple_barrier, kwargs=params)
        )

        # Compute expected win/loss for kelly criterion bet sizing
        secondary_target_params = self.freqai_info["secondary_target_parameters"]
        window = secondary_target_params["tmb_long_window"]

        # Compute trade return when TBM barrier crossing occurs
        dataframe["!-trade_return_long"] = (
                (dataframe["primary_enter_long_tbm_value"] - dataframe["close"]) /
                dataframe["close"]
        )
        dataframe["!-trade_return_short"] = (
                (dataframe["primary_enter_short_tbm_value"] - dataframe["close"]) /
                dataframe["close"]
        )

        # Mask down to long/short entry triggers only (above proba threshold)
        dataframe["!-trade_return_long_masked"] = np.where(
            (dataframe["primary_enter_long"] == True),
            dataframe["!-trade_return_long"], np.nan)
        dataframe["!-trade_return_short_masked"] = np.where(
            (dataframe["primary_enter_short"] == True),
            dataframe["!-trade_return_short"], np.nan)

        # Compute expected win for long/short trades
        dataframe["!-trade_return_long_expected_win"] = dataframe.loc[
            dataframe["!-trade_return_long_masked"] > 0, "!-trade_return_long_masked"].rolling(
            1000, min_periods=1).mean()
        dataframe["!-trade_return_long_expected_win"] = dataframe[
            "!-trade_return_long_expected_win"].fillna(method="backfill").fillna(method="ffill")

        dataframe["!-trade_return_long_expected_loss"] = dataframe.loc[
            dataframe["!-trade_return_long_masked"] <= 0, "!-trade_return_long_masked"].rolling(
            1000, min_periods=1).mean()
        dataframe["!-trade_return_long_expected_loss"] = dataframe[
            "!-trade_return_long_expected_loss"].fillna(method="backfill").fillna(method="ffill")

        dataframe["!-trade_return_short_expected_win"] = dataframe.loc[
            dataframe["!-trade_return_short_masked"] <= 0, "!-trade_return_short_masked"].rolling(
            1000, min_periods=1).mean()
        dataframe["!-trade_return_short_expected_win"] = dataframe[
            "!-trade_return_short_expected_win"].fillna(method="backfill").fillna(method="ffill")

        dataframe["!-trade_return_short_expected_loss"] = dataframe.loc[
            dataframe["!-trade_return_short_masked"] > 0, "!-trade_return_short_masked"].rolling(
            1000, min_periods=1).mean()
        dataframe["!-trade_return_short_expected_loss"] = dataframe[
            "!-trade_return_short_expected_loss"].fillna(method="backfill").fillna(method="ffill")

        # Simple trade return measure for sample_weights
        dataframe["!-trade_return"] = dataframe["close"].pct_change(window).shift(-window)

        # Compute median max price change within window to enable auto TBM threshold setting
        dataframe["!-trade_return_max_price_change"] = ((
                                                            dataframe["high"]
                                                            .shift(-window)
                                                            .rolling(window + 1)
                                                            .max()
                                                        ) - dataframe["close"]) / dataframe["close"]

        dataframe["!-trade_return_min_price_change"] = (
                                                               dataframe["low"] - dataframe["close"]
                                                               .shift(-window)
                                                               .rolling(window + 1)
                                                               .min()
                                                       ) / dataframe["close"]

        # Mask down to long/short entry triggers only
        dataframe["!-trade_return_max_price_change_masked"] = np.where(
            dataframe["primary_enter_long"], dataframe["!-trade_return_max_price_change"], np.nan)
        dataframe["!-trade_return_min_price_change_masked"] = np.where(
            dataframe["primary_enter_short"], dataframe["!-trade_return_min_price_change"], np.nan)

        # Compute rolling median over masked data
        dataframe["!-trade_return_median_max_price_change"] = dataframe.loc[
            dataframe["!-trade_return_max_price_change_masked"] > 0,
            "!-trade_return_max_price_change_masked"].rolling(
            5000, min_periods=1).median()
        dataframe["!-trade_return_median_max_price_change"] = dataframe[
            "!-trade_return_median_max_price_change"].fillna(
            method="backfill").fillna(method="ffill")

        dataframe["!-trade_return_median_min_price_change"] = dataframe.loc[
            dataframe["!-trade_return_min_price_change_masked"] > 0,
            "!-trade_return_min_price_change_masked"].rolling(
            5000, min_periods=1).median()
        dataframe["!-trade_return_median_min_price_change"] = dataframe[
            "!-trade_return_median_min_price_change"].fillna(
            method="backfill").fillna(method="ffill")

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

        # Long
        long_tbm_map = {1: "win_long", 0: "loss_long", -1: "loss_long"}
        dataframe["long_outcome_tbm"] = dataframe["primary_enter_long_tbm"].map(long_tbm_map)

        conditions = [dataframe["primary_enter_long"]]
        choices = [dataframe["long_outcome_tbm"]]
        dataframe["&-meta_target_binary_long"] = np.select(conditions, choices, default="drop-row")
        dataframe["&-meta_target_binary_long"] = dataframe["&-meta_target_binary_long"].fillna(
            value="drop-row")

        # Short
        short_tbm_map = {1: "loss_short", 0: "loss_short", -1: "win_short"}
        dataframe["short_outcome_tbm"] = dataframe["primary_enter_short_tbm"].map(short_tbm_map)

        conditions = [dataframe["primary_enter_short"]]
        choices = [dataframe["short_outcome_tbm"]]
        dataframe["&-meta_target_binary_short"] = np.select(conditions, choices, default="drop-row")
        dataframe["&-meta_target_binary_short"] = dataframe["&-meta_target_binary_short"].fillna(
            value="drop-row")

        print(dataframe.groupby("&-meta_target_binary_long").size())
        print(dataframe.groupby("&-meta_target_binary_short").size())

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        # Trigger thresholds
        smoothing_window = self.freqai_info["entry_parameters"].get("smoothing_window", 30)

        dataframe["win_long_enter_threshold"] = dataframe[
            "win_long_threshold_&-meta_target_binary_long"].rolling(smoothing_window).mean()

        dataframe["win_short_enter_threshold"] = dataframe[
            "win_short_threshold_&-meta_target_binary_short"].rolling(smoothing_window).mean()

        # Indicator to confirm ML model trained
        dataframe["long_model_ready"] = (dataframe["win_long"].mean() != 0)
        dataframe["short_model_ready"] = (dataframe["win_short"].mean() != 0)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> pd.DataFrame:

        # Long Entry
        conditions = [df["primary_enter_long"],
                      df["long_model_ready"],
                      df["win_long"] >= df["win_long_enter_threshold"]]

        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "meta_enter_long")

        # Short Entry
        conditions = [df["primary_enter_short"],
                      df["short_model_ready"],
                      df["win_short"] >= df["win_short_enter_threshold"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "meta_enter_short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> pd.DataFrame:
        # Use custom_exit() instead for more control
        return df

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        exit_params = self.freqai_info["exit_parameters"]

        weak_profit_threshold = 0.005

        # When not in strong profit, exit when weak opposing entry detected
        if exit_params["weak_opposing_entry_trigger_enabled"]:
            if 0 < current_profit < weak_profit_threshold:
                if ((last_candle["primary_enter_short"] == True) and
                        (trade.trade_direction == "long")):
                    return "exit_long_weak_opposing_signal"

                elif ((last_candle["primary_enter_long"] == True) and
                      (trade.trade_direction == "short")):
                    return "exit_short_weak_opposing_signal"

        # When in strong profit, only exit when strong opposing entry detected
        if exit_params["strong_opposing_entry_trigger_enabled"]:
            if weak_profit_threshold < current_profit:
                if ((last_candle["primary_enter_short"] == True) and
                        (last_candle["short_model_ready"] == True) and
                        (last_candle["win_short"] >= last_candle["win_short_enter_threshold"]) and
                        (trade.trade_direction == "long")):
                    return "exit_long_strong_opposing_signal"

                elif ((last_candle["primary_enter_long"] == True) and
                      (last_candle["long_model_ready"] == True) and
                      (last_candle["win_long"] >= last_candle["win_long_enter_threshold"]) and
                      (trade.trade_direction == "short")):
                    return "exit_short_strong_opposing_signal"

        # When in strong profit, only exit when strong opposing entry detected
        if exit_params["exit_trigger_enabled"]:
            if ((last_candle["primary_exit_long"] == True) and
                    (trade.trade_direction == "long")):
                return "exit_long_signal"

            elif ((last_candle["primary_exit_short"] == True) and
                  (trade.trade_direction == "short")):
                return "exit_short_signal"

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        bet_sizing_params = self.freqai_info["bet_sizing"]
        if bet_sizing_params["enabled"]:

            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()

            # Get available balance to bet
            total_stake_amount = max_stake

            # Calculate kelly criterion bet size
            if side == "long":
                pred_proba = current_candle["win_long"]
                expected_win = current_candle["!-trade_return_long_expected_win"]
                expected_loss = current_candle["!-trade_return_long_expected_loss"]
                print(current_candle[["win_long", "!-trade_return_long_expected_win",
                                      "!-trade_return_long_expected_loss"]])
            else:
                pred_proba = current_candle["win_short"]
                expected_win = current_candle["!-trade_return_short_expected_win"]
                expected_loss = current_candle["!-trade_return_short_expected_loss"]
                print(current_candle[["win_short", "!-trade_return_short_expected_win",
                                      "!-trade_return_short_expected_loss"]])

            kelly_fraction = 0.2
            kelly_factor = bet_sizing.kelly_bet_size(
                p=pred_proba,
                win=expected_win,
                loss=expected_loss,
                kelly_fraction=kelly_fraction
            )

            bet_amount = kelly_factor * total_stake_amount

            self.dp.send_msg("*Kelly Criterion Bet Sizing:* \n"
                             f"*Pair:* {pair} \n"
                             f"*Side:* {side} \n"
                             f"*Total available funds:* {np.round(total_stake_amount, 4)} \n"
                             f"*Bet amount:* {np.round(bet_amount, 4)} \n"
                             f"*Kelly factor:* {kelly_factor} \n"
                             f"*Prediction proba:* {np.round(pred_proba, 4)} \n"
                             f"*Expected win:* {np.round(expected_win, 4)} \n"
                             f"*Expected loss:* {np.round(expected_loss, 4)} \n"
                             f"*Kelly fraction:* {kelly_fraction}")

            return bet_amount

        else:
            return proposed_stake

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

        if current_profit <= 0.00:
            return -1

        elif current_profit > 0.005:
            return 0.025

        elif current_profit > 0.01:
            desired_stoploss = current_profit

            min_sl = 0.01
            max_sl = 0.02

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
        if ((current_time - trade.date_last_filled_utc > timedelta(minutes=15)) &
                (trade.nr_of_successful_entries <= self.max_entry_position_adjustment) &
                (current_profit < -0.005)):

            df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            last_candle = df.iloc[-1].squeeze()

            # Long Entry
            enter_long = np.where(
                ((last_candle["primary_enter_long"] == True) and
                 (last_candle["long_model_ready"] == True) and
                 (last_candle["win_long"] >= last_candle["win_long_enter_threshold"])),
                True, False)

            # Short Entry
            enter_short = np.where(
                ((last_candle["primary_enter_short"] == True) &
                 (last_candle["short_model_ready"] == True) &
                 (last_candle["win_short"] >= last_candle["win_short_enter_threshold"])),
                True, False)

            if enter_long or enter_short:
                # This returns first order stake size
                filled_entries = trade.select_filled_orders(trade.entry_side)
                stake_amount = filled_entries[0].cost
                logger.info(f"Trade adjustment made adding {stake_amount} to {trade.pair}")
                return stake_amount / 3.0

        return None
