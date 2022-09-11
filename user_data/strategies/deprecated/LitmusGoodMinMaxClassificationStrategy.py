import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import time

# from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from functools import reduce
from pandas import DataFrame
from scipy.signal import find_peaks
from technical import qtpylib

# from user_data.litmus import label_helpers

logger = logging.getLogger(__name__)


class LitmusGoodMinMaxClassificationStrategy(IStrategy):

    minimal_roi = {"0": 0.1, "600": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
                "DI_values": {"color": "grey"},
            },
            "Long": {
                "good_entry_long": {"color": "SpringGreen"},
                "good_exit_long": {"color": "MediumVioletRed"},
            },
            "Short": {
                "good_entry_short": {"color": "SpringGreen"},
                "good_exit_short": {"color": "MediumVioletRed"},
            },
            "True": {
                "true_good_entry_long": {"color": "#4dff4d"},
                "true_good_exit_long": {"color": "#ff471a"},
                "true_bad_entry_long": {"color": "#e6ffe6"},
                "true_bad_exit_long": {"color": "#ffebe6"},
            },
            "Prec": {
                "pr_auc_is_minima": {"color": "Thistle"},
                "pr_auc_is_maxima": {"color": "SteelBlue"},
                "pr_auc_missed_minima": {"color": "Wheat"},
                "pr_auc_missed_maxima": {"color": "Plum"}
            },
        },
    }

    # Stop loss config
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 300
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

            # Find all good entries and exits for longs
            min_duration = 60
            min_growth = 0.02

            long_labels = entry_exit_labeler(
                df["close"], min_growth=min_growth, max_duration=min_duration, suffix="_long"
            )
            """short_labels = label_helpers.entry_exit_labeler(
                df["close"], min_growth=min_growth, max_duration=min_duration, suffix="_short"
            )"""

            df = pd.concat([df, long_labels], axis=1)

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)

        # enter_mul = 2.6
        # exit_mul = 1.7
        trigger_window = 300
        print(dataframe.columns)

        dataframe["test"] = dataframe["close"].rolling(trigger_window).mean()
        """dataframe["is_maxima_mean"] = dataframe["is_maxima"].rolling(trigger_window).mean()
        dataframe["is_maxima_std"] = dataframe["is_maxima"].rolling(trigger_window).std()
        dataframe["is_minima_mean"] = dataframe["is_minima"].rolling(trigger_window).mean()
        dataframe["is_minima_std"] = dataframe["is_minima"].rolling(trigger_window).std()

        dataframe["missed_maxima_mean"] = dataframe["missed_maxima"].rolling(trigger_window).mean()
        dataframe["missed_maxima_std"] = dataframe["missed_maxima"].rolling(trigger_window).std()
        dataframe["missed_minima_mean"] = dataframe["missed_minima"].rolling(trigger_window).mean()
        dataframe["missed_minima_std"] = dataframe["missed_minima"].rolling(trigger_window).std()

        # Short
        dataframe["short_entry_target"] = (
                dataframe["is_maxima_mean"] + dataframe["is_maxima_std"] * enter_mul)
        dataframe["short_exit_target"] = (
                dataframe["is_minima_mean"] + dataframe["is_minima_std"] * exit_mul)
        dataframe["missed_short_entry_target"] = (
                dataframe["missed_maxima_mean"] + dataframe["missed_maxima_std"] * enter_mul)
        dataframe["missed_short_exit_target"] = (
                dataframe["missed_minima_mean"] + dataframe["missed_minima_std"] * exit_mul)

        # Long
        dataframe["long_entry_target"] = (
                dataframe["is_minima_mean"] + dataframe["is_minima_std"] * enter_mul)
        dataframe["long_exit_target"] = (
                dataframe["is_maxima_mean"] + dataframe["is_maxima_std"] * exit_mul)
        dataframe["missed_long_entry_target"] = (
                dataframe["missed_minima_mean"] + dataframe["missed_minima_std"] * enter_mul)
        dataframe["missed_long_exit_target"] = (
                dataframe["missed_maxima_mean"] + dataframe["missed_maxima_std"] * exit_mul)"""

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Long Entry
        conditions = [df["do_predict"] == 1, df["test"] > 0]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "is_minima")

        # Short Entry
        conditions = [df["do_predict"] == 1, df["test"] > 0]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "is_maxima")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Long Exit
        conditions = [1 == 1, df["test"] > 0]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]
            ] = (1, "is_maxima")

        # Short Exit
        conditions = [1 == 1, df["test"] > 0]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_short", "exit_tag"]
            ] = (1, "is_minima")

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


def minmax_growth_np(ts):
    """Just for testing without numba... very slow for large ts"""
    ts = ts.astype(np.float32)
    res = np.empty((ts.shape[0], ts.shape[0]), dtype=ts.dtype)
    for i in range(ts.shape[0]):
        for j in range(ts.shape[0]):
            r = (ts[j] - ts[i]) / ts[i]
            res[i, j] = r
    return res


def entry_exit_labeler(close_df, min_growth, max_duration, suffix=None):
    """Label timeseries for good entries and exits at peaks"""

    logger.info(f"Labeling for {len(close_df)} samples")
    logger.info(f"Labeling settings: min_growth {min_growth}, max_duration {max_duration}")
    start_time = time.time()

    close = close_df.values

    # Find all peaks and valleys
    exit_idx = find_peaks(close)[0]
    entry_idx = find_peaks(-close)[0]

    # Mask valleys & peaks
    min_mask = np.zeros(close.shape, dtype=bool)
    min_mask[entry_idx] = True
    max_mask = np.zeros(close.shape, dtype=bool)
    max_mask[exit_idx] = True

    # Compute distance matrix between all close points
    dist_matrix = minmax_growth_np(close)

    # Scope dist_matrix to distances > min_threshold
    growth_mask = (dist_matrix - min_growth) > 0
    dist_matrix = dist_matrix * growth_mask

    # Limit to max_duration & remove lower triangle
    dist_matrix = np.tril(dist_matrix, k=max_duration)
    dist_matrix = np.triu(dist_matrix)

    # Scope to only entries & get idx
    dist_matrix = dist_matrix * min_mask.reshape(-1, 1)
    good_entry_idx, _ = np.where(dist_matrix > 0)

    # Scope to entries & exits & find idx
    dist_matrix = dist_matrix * max_mask
    _, good_exit_idx = np.where(dist_matrix > 0)

    # Populate DataFrame and return
    col_names = ["&target", "true_good_entry", "true_good_exit", "true_bad_entry", "true_bad_exit"]
    col_names = [i + suffix for i in col_names]
    label_names = ["good_entry", "good_exit", "bad_entry", "bad_exit", "transition"]
    label_names = [i + suffix for i in label_names]

    df = pd.DataFrame(columns=col_names, index=close_df.index)
    df[col_names[1:]] = 0

    # Bad entries
    df.loc[[i for i in entry_idx if i not in good_entry_idx], col_names[0]] = label_names[2]
    df.loc[[i for i in entry_idx if i not in good_entry_idx], col_names[3]] = 1
    # Good entries
    df.loc[good_entry_idx, col_names[0]] = label_names[0]
    df.loc[good_entry_idx, col_names[1]] = 1
    # Bad exit
    df.loc[[i for i in exit_idx if i not in good_exit_idx], col_names[0]] = label_names[3]
    df.loc[[i for i in exit_idx if i not in good_exit_idx], col_names[4]] = 1
    # Good exit
    df.loc[good_exit_idx, col_names[0]] = label_names[1]
    df.loc[good_exit_idx, col_names[2]] = 1
    # Fill the rest with filler_value
    df[col_names[0]].fillna(value=label_names[4], inplace=True)

    end_time = time.time() - start_time
    logger.info(f"Time taken to label data: {end_time} seconds")
    print(df)

    return df
