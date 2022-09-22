from freqtrade.strategy import IStrategy, merge_informative_pair
from functools import reduce
from pandas import DataFrame
from technical import qtpylib

import logging
import pandas as pd
import talib.abstract as ta
import zigzag

logger = logging.getLogger(__name__)


class LitmusMinMaxClassificationStrategy(IStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusMinMaxClassificationStrategy
      --config user_data/strategies/config.LitmusMinMaxClassification.json
      --freqaimodel LitmusMultiTargetClassifier --verbose
    """

    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {
                "do_predict": {"color": "brown"},
                "DI_values": {"color": "grey"},
            },
            "Long": {
                "missed2_minima": {"color": "PaleGreen"},
                "missed1_minima": {"color": "ForestGreen"},
                "missed_long_entry_target": {"color": "ForestGreen"},
                "missed2_maxima": {"color": "Salmon"},
                "missed1_maxima": {"color": "Crimson"},
                "missed_long_exit_target": {"color": "Crimson"},
            },
            "Short": {
                "missed2_maxima": {"color": "PaleGreen"},
                "missed1_maxima": {"color": "ForestGreen"},
                "missed_short_entry_target": {"color": "ForestGreen"},
                "missed2_minima": {"color": "Salmon"},
                "missed1_minima": {"color": "Crimson"},
                "missed_short_exit_target": {"color": "Crimson"},
            },
            "Labels": {
                "real_peaks": {"color": "Blue"}
            },
            "Time": {
                "time_to_train": {"color": "DarkGray"}
            },
        },
    }

    # Stop loss config
    stoploss = -0.03
    """trailing_stop = True
    trailing_stop_positive_offset = 0.01
    trailing_stop_positive = 0.005
    trailing_only_offset_is_reached = True"""

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 300
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

        # informative[f"%-{coin}-pct-change"] = informative["close"].pct_change()
        # informative[f"%-{coin}-raw_volume"] = informative["volume"]
        # informative[f"%-{coin}-raw_price"] = informative["close"]

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

            # Zigzag min/max for long pivot positions
            min_growth = self.freqai_info["labeling_parameters"].get(
                "min_growth", -1)
            min_retraction = self.freqai_info["labeling_parameters"].get(
                "min_retraction", -1)
            peaks = zigzag.peak_valley_pivots(
                df["close"].values, min_growth, -min_retraction)
            name_map = {0: "not_minmax", 1: "maxima", -1: "minima",
                        2: "missed1_maxima", -2: "missed1_minima",
                        3: "missed2_maxima", -3: "missed2_minima"}
            peaks[0] = 0  # Set first value of peaks = 0
            peaks[-1] = 0  # Set last value of peaks = 0
            df["&target"] = peaks
            df["&target"] = df["&target"].map(name_map)
            df["real_peaks"] = peaks

            # Missed entries & exits (labels)
            df.loc[(df["&target"].shift(1) == name_map[1]), "&target"] = name_map[2]
            df.loc[(df["&target"].shift(1) == name_map[-1]), "&target"] = name_map[-2]

            df.loc[(df["&target"].shift(2) == name_map[2]), "&target"] = name_map[3]
            df.loc[(df["&target"].shift(2) == name_map[-2]), "&target"] = name_map[-3]

            # Reset minima / maxima label back to not_minmax (predictions not used)
            df.loc[(df["&target"] == name_map[1]), "&target"] = name_map[0]
            df.loc[(df["&target"] == name_map[-1]), "&target"] = name_map[0]

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)

        enter_mul = 2.6
        exit_mul = 1.7

        # Long entry targets
        dataframe["missed_long_entry_target"] = (
            dataframe["missed1_minima_mean"] + dataframe["missed1_minima_std"] * enter_mul)

        # Long exit targets
        dataframe["missed_long_exit_target"] = (
            dataframe["missed1_maxima_mean"] + dataframe["missed1_maxima_std"] * exit_mul)

        # Short exit targets
        dataframe["missed_short_exit_target"] = (
                dataframe["missed1_minima_mean"] + dataframe["missed1_minima_std"] * exit_mul)

        # Short entry targets
        dataframe["missed_short_entry_target"] = (
                dataframe["missed1_maxima_mean"] + dataframe["missed1_maxima_std"] * enter_mul)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Missed Long Entry
        conditions = [
            1 == 1,
            qtpylib.crossed_above(df["missed1_minima"], df["missed_long_entry_target"])]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "missed1_minima")

        # Missed Short Entry
        conditions = [
            1 == 1,
            qtpylib.crossed_above(df["missed1_maxima"], df["missed_short_entry_target"])]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "missed1_maxima")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Long Exit
        conditions = [
            1 == 1,
            qtpylib.crossed_above(df["missed1_maxima"], df["missed_long_exit_target"])]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]
            ] = (1, "missed1_maxima")

        # Short Exit
        conditions = [
            1 == 1,
            qtpylib.crossed_above(df["missed1_minima"], df["missed_short_exit_target"])]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_short", "exit_tag"]
            ] = (1, "missed1_minima")

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    """def custom_exit(
            self,
            pair: str,
            trade: Trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            **kwargs
    ):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        trade_date = timeframe_to_prev_date(
            self.timeframe, (trade.open_date_utc - timedelta(minutes=int(self.timeframe[:-1])))
        )
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        trade_df = dataframe.loc[(dataframe["date"] > trade_date)]
        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()
        entry_tag = trade.enter_tag

        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if trade_duration > 500:
            return 'trade expired'

        if trade_df['maxima-exit'].sum() > 0 and entry_tag == 'long':
            return 'maxima_missed'

        if trade_df['minima-exit'].sum() > 0 and entry_tag == 'short':
            return 'minima_missed'

        if current_profit < -0.07:
            return "stop_loss"

        if last_candle['minima'] == 1 and entry_tag == 'short':
            return "minimia_detected_short"

        if last_candle['maxima'] == 1 and entry_tag == 'long':
            return "maxima_detected_long"
            """

    """def confirm_trade_entry(
            self,
            pair: str,
            order_type: str,
            amount: float,
            rate: float,
            time_in_force: str,
            current_time: datetime,
            entry_tag: Optional[str],
            side: str,
            **kwargs
    ) -> bool:

        open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))

        num_shorts, num_longs = 0, 0
        for trade in open_trades:
            if trade.enter_tag == 'short':
                num_shorts += 1
            elif trade.enter_tag == 'long':
                num_longs += 1

        if side == "long" and num_longs >= 4:
            return False

        if side == "short" and num_shorts >= 4:
            return False

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
        """

    """def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        bid = self.wallets.get_available_stake_amount() * current_candle["missed_long_entry"]

        return bid
        """
