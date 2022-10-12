from freqtrade.strategy import IStrategy, merge_informative_pair
from functools import reduce
from pandas import DataFrame
from technical import qtpylib

import logging
import pandas as pd
import talib.abstract as ta
import zigzag

logger = logging.getLogger(__name__)


class LitmusMinMaxSegmentClassificationStrategy(IStrategy):
    """
    to run this:
      freqtrade trade --strategy LitmusMinMaxSegmentClassificationStrategy
      --config user_data/strategies/config.LitmusMinMaxSegmentClassification.json
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
            "Segments": {
                "entry_threshold_long": {"color": "DarkGray"},
                "entry_threshold_short": {"color": "DarkGray"},
                "exit_threshold_long": {"color": "DarkGray"},
                "exit_threshold_short": {"color": "DarkGray"},
                "delta_segment_1x1x": {"color": "#4285F4"}
            },
            "1X1X": {
                "long_segment_1x1x": {"color": "ForestGreen"},
                "short_segment_1x1x": {"color": "Crimson"}
            },
            "Labels": {
                "real_segment_1x1x": {"color": "#4285F4"}
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
        """try:
            print(df["%-long_entry_pred"].columns)
        except:
            pass"""

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
        informative[f"%-{coin}-raw_volume"] = informative["volume"]
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

            # Segments: Long 1X Short 1X
            min_growth_long = self.freqai_info["labeling_parameters"].get(
                "min_growth_long", -1)
            min_retraction_long = self.freqai_info["labeling_parameters"].get(
                "min_retraction_long", -1)
            peak_1x1x = zigzag.peak_valley_pivots(
                df["close"].values, min_growth_long, -min_retraction_long)
            segment_1x1x = zigzag.pivots_to_modes(peak_1x1x)
            df["&segment_1x1x"] = segment_1x1x
            df["&segment_1x1x"] = df["&segment_1x1x"].map(
                {1: "long_segment_1x1x", -1: "short_segment_1x1x"})
            df["real_segment_1x1x"] = peak_1x1x

            # Set start and end as not peaks
            # TODO: Beginning and end of seq might not be correctly labeled
            """long_peaks[0] = 0  # Set first value of peaks = 0
            long_peaks[-1] = 0  # Set last value of peaks = 0"""

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)

        # Entry/Exit thresholds
        dataframe["entry_threshold_long"] = 0.2
        dataframe["entry_threshold_short"] = -0.2
        dataframe["exit_threshold_long"] = 0
        dataframe["exit_threshold_short"] = 0

        # Delta Segment Predictions
        dataframe["delta_segment_1x1x"] = (
                dataframe["long_segment_1x1x"] - dataframe["short_segment_1x1x"])

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Delta Segment Long Entry
        conditions = [
            (df["delta_segment_1x1x"].shift(1) > 0),
            (df["delta_segment_1x1x"] > df["entry_threshold_long"])]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]
            ] = (1, "segment_1x1x_long")

        # Delta Segment Short Entry
        conditions = [
            (df["delta_segment_1x1x"].shift(1) < 0),
            (df["delta_segment_1x1x"] < df["entry_threshold_short"])]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["enter_short", "enter_tag"]
            ] = (1, "segment_1x1x_short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # Delta Segment Long Exit
        conditions = [
            df["delta_segment_1x1x"].shift(1) > df["delta_segment_1x1x"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]
            ] = (1, "segment_1x1x_long")

        # Delta Segment Short Exit
        conditions = [
            df["delta_segment_1x1x"].shift(1) < df["delta_segment_1x1x"]]
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions), ["exit_short", "exit_tag"]
            ] = (1, "segment_1x1x_short")

        return df

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])

    """def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        bid = self.wallets.get_available_stake_amount() * current_candle["missed_long_entry"]

        return bid"""
