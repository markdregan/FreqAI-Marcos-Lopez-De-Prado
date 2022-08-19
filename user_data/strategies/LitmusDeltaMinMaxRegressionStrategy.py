
# from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from functools import reduce
from pandas import DataFrame
from scipy.signal import argrelextrema
from technical import qtpylib

import logging
import numpy as np
import pandas as pd
import talib.abstract as ta

logger = logging.getLogger(__name__)


class LitmusDeltaMinMaxRegressionStrategy(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.freqai.start(dataframe, metadata)
    to make predictions on their data. populate_any_indicators() automatically
    generates the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    """

    minimal_roi = {"0": 0.1, "600": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
                "DI_values": {"color": "grey"},
            },
            "Long": {
                "&delta_minmax": {"color": "CornflowerBlue"},
                "long_exit_target": {"color": "FireBrick"},
                "long_entry_target": {"color": "DarkOliveGreen"},
            },
            "Short": {
                "&delta_minmax": {"color": "CornflowerBlue"},
                "short_entry_target": {"color": "DarkOliveGreen"},
                "short_exit_target": {"color": "FireBrick"},
            },
            "Real": {
                "real_minima": {"color": "blue"},
                "real_maxima": {"color": "yellow"}
            },
            "Extra": {
                "delta_minmax_plot": {"color": "Thistle"}
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = True

    linear_roi_offset = DecimalParameter(
        0.00, 0.02, default=0.005, space="sell", optimize=False, load=True
    )
    max_roi_time_long = IntParameter(0, 800, default=400, space="sell", optimize=False, load=True)

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
            informative[f"%-{coin}-rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{coin}-mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{coin}-adx-period_{t}"] = ta.ADX(informative, window=t)
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
            df["%-day_of_week"] = df["date"].dt.dayofweek
            df["%-hour_of_day"] = df["date"].dt.hour

            # Define Min & Max binary indicators
            min_peaks = argrelextrema(df["close"].values, np.less, order=15)[0]
            max_peaks = argrelextrema(df["close"].values, np.greater, order=15)[0]

            df["real_minima"] = 0
            df["real_maxima"] = 0
            df.loc[min_peaks, "real_minima"] = 1
            df.loc[max_peaks, "real_maxima"] = 1

            df["next_peak_close"] = np.nan
            df.loc[min_peaks, "next_peak_close"] = df.loc[min_peaks, "close"]
            df.loc[max_peaks, "next_peak_close"] = df.loc[max_peaks, "close"]
            df["next_peak_close"].fillna(method="backfill", axis=0, inplace=True)

            # Delta to Next Min/Max Regression Target
            df["&delta_minmax"] = df["next_peak_close"].shift(-1) / df["close"] - 1
            df["delta_minmax_plot"] = df["&delta_minmax"]

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        # All indicators must be populated by populate_any_indicators() for live functionality
        # to work correctly.

        # the model will return all labels created by user in `populate_any_indicators`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `populate_any_indicators()` for each training period.

        dataframe = self.freqai.start(dataframe, metadata, self)

        enter_mul = 1.5
        exit_mul = 0.2
        trigger_window = 300

        print(dataframe.columns)

        dataframe["delta_minmax_mean"] = dataframe["&delta_minmax"].rolling(trigger_window).mean()
        dataframe["delta_minmax_std"] = dataframe["&delta_minmax"].rolling(trigger_window).std()

        # Short
        dataframe["short_entry_target"] = (
                dataframe["delta_minmax_mean"] - dataframe["delta_minmax_std"] * enter_mul)
        dataframe["short_exit_target"] = (
                dataframe["delta_minmax_mean"] - dataframe["delta_minmax_std"] * exit_mul)
        """dataframe["missed_short_entry_target"] = (
                dataframe["missed_maxima_mean"] + dataframe["missed_maxima_std"] * enter_mul)
        dataframe["missed_short_exit_target"] = (
                dataframe["missed_minima_mean"] + dataframe["missed_minima_std"] * exit_mul)"""

        # Long
        dataframe["long_entry_target"] = (
                dataframe["delta_minmax_mean"] + dataframe["delta_minmax_std"] * enter_mul)
        dataframe["long_exit_target"] = (
                dataframe["delta_minmax_mean"] + dataframe["delta_minmax_std"] * exit_mul)
        """dataframe["missed_long_entry_target"] = (
                dataframe["missed_minima_mean"] + dataframe["missed_minima_std"] * enter_mul)
        dataframe["missed_long_exit_target"] = (
                dataframe["missed_maxima_mean"] + dataframe["missed_maxima_std"] * exit_mul)"""

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Long Entry
        conditions = [df["do_predict"] == 1, df["&delta_minmax"] > df["long_entry_target"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "is_minima")

        """# Missed Long Entry
        conditions = [df["do_predict"] == 1, df["missed_minima"] > df["missed_long_entry_target"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "missed_minima")"""

        # Short Entry
        conditions = [df["do_predict"] == 1, df["&delta_minmax"] < df["short_entry_target"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "is_maxima")

        """# Missed Short Entry
        conditions = [df["do_predict"] == 1, df["missed_maxima"] > df["missed_short_entry_target"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "missed_maxima")"""

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Long Exit
        conditions = [1 == 1, df["&delta_minmax"] < df["long_exit_target"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]
            ] = (1, "is_maxima")

        """# Missed Long Exit
        conditions = [1 == 1, df["missed_maxima"] > df["missed_long_exit_target"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]
            ] = (1, "missed_maxima")"""

        # Short Exit
        conditions = [1 == 1, df["&delta_minmax"] > df["short_exit_target"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_short", "exit_tag"]
            ] = (1, "is_minima")

        """# Missed Short Exit
        conditions = [1 == 1, df["missed_minima"] > df["missed_short_exit_target"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_short", "exit_tag"]
            ] = (1, "missed_minima")"""

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    """def custom_exit(
        self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        trade_date = timeframe_to_prev_date(self.config["timeframe"], trade.open_date_utc)
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()
        follow_mode = self.config.get("freqai", {}).get("follow_mode", False)
        if not follow_mode:
            pair_dict = self.freqai.dd.pair_dict
        else:
            pair_dict = self.freqai.dd.follower_dict
        entry_tag = trade.enter_tag
        if (
            "prediction" + entry_tag not in pair_dict[pair]
            or pair_dict[pair]["prediction" + entry_tag] > 0
        ):
            with self.freqai.lock:
                pair_dict[pair]["prediction" + entry_tag] = abs(trade_candle["&-s_close"])
                if not follow_mode:
                    self.freqai.dd.save_drawer_to_disk()
                else:
                    self.freqai.dd.save_follower_dict_to_disk()
        roi_price = pair_dict[pair]["prediction" + entry_tag]
        roi_time = self.max_roi_time_long.value
        roi_decay = roi_price * (
            1 - ((current_time - trade.open_date_utc).seconds) / (roi_time * 60)
        )
        if roi_decay < 0:
            roi_decay = self.linear_roi_offset.value
        else:
            roi_decay += self.linear_roi_offset.value
        if current_profit > roi_decay:
            return "roi_custom_win"
        if current_profit < -roi_decay:
            return "roi_custom_loss"
            """

    """def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time,
        **kwargs,
    ) -> bool:

        entry_tag = trade.enter_tag
        follow_mode = self.config.get("freqai", {}).get("follow_mode", False)
        if not follow_mode:
            pair_dict = self.freqai.dd.pair_dict
        else:
            pair_dict = self.freqai.dd.follower_dict

        with self.freqai.lock:
            pair_dict[pair]["prediction" + entry_tag] = 0
            if not follow_mode:
                self.freqai.dd.save_drawer_to_disk()
            else:
                self.freqai.dd.save_follower_dict_to_disk()

        return True"""

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
