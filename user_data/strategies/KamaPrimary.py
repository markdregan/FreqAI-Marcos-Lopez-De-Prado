# KAMA Primary Strategy
# Tradingview: https://www.tradingview.com/chart/WeEVLg4V/
# Author: markdregan@gmail.com

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
import numpy as np
from pandas import DataFrame
from ta import add_all_ta_features
from ta.momentum import KAMAIndicator
from user_data.litmus import indicator_helpers, external_informative_data as eid

import freqtrade.vendor.qtpylib.indicators as qtpylib

# Prevent pandas complaining with future warning errors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

master_plot_config = {
    "main_plot": {
        "kama": {
            "color": "purple"
        },
    },
    "subplots": {
        "KAMA": {
            "kama_entry_threshold": {
                "color": "green"
            },
            "kama_exit_threshold": {
                "color": "red"
            },
            "kama_delta": {
                "color": "purple"
            },
        },
        "POS": {
            "kama_entry_pos": {
                "color": "green"
            },
            "kama_exit_pos": {
                "color": "red"
            },
        },
    },
}


class KamaPrimary(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "5m"

    # Required. But effectively disabled.
    minimal_roi = {
        # Effectively disabled. Not used for MetaModel. PT & SL levels defined in custom_sell()
        '1000': 1000
    }

    # Stop loss disabled. Use custom_exit() for TBM meta labeling.
    stoploss = -1

    # Turn on position adjustment
    position_adjustment_enable = False

    # Plotting config
    plot_config = master_plot_config

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count = 100

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign timeframe to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        # Add additional informative pairs
        informative_pairs += [
            ("BTC/USDT", "1h"),
            ("BTC/USDT", "5m")
        ]

        return informative_pairs

    def bot_loop_start(self, **kwargs) -> None:

        # Informative: Glassnode 1d
        glassnode_df = eid.load_local_data('glassnode_BTC_1d.csv', 'glassnode')
        self.dataframe_glassnode_1d = glassnode_df  # TODO: Add fracdiff in pipeline

        # Informative: BTC 5m
        self.dataframe_btc_5m = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe='5m')
        self.dataframe_btc_5m = indicator_helpers.add_ta_informative(
            self.dataframe_btc_5m, suffix='_btc')

        # Informative: BTC 1h
        self.dataframe_btc_1h = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe='1h')
        self.dataframe_btc_1h = indicator_helpers.add_ta_informative(
            self.dataframe_btc_1h, suffix='_btc')

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:

        # Apply TA indicators to the pair dataframe
        dataframe = add_all_ta_features(
            dataframe, open="open", high="high", low="low",
            close="close", volume="volume",  fillna=True)

        # Informative signals of pair at lower time resolution
        low_res_tf = '1h'
        dataframe_low_res = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=low_res_tf)
        dataframe_low_res = indicator_helpers.add_ta_informative(
            dataframe_low_res, suffix='')

        # KAMA
        kama_window = 14
        kama = KAMAIndicator(dataframe['close'], window=kama_window, pow1=2, pow2=20)
        dataframe['kama'] = kama.kama()
        dataframe['kama_delta'] = dataframe['kama'] - dataframe['kama'].shift(1)

        # Entry/Exit dynamic threshold
        kama_entry_coeff = 1
        kamma_exit_coeff = -0.5
        dataframe['kama_threshold'] = dataframe['kama_delta'].rolling(window=kama_window).std()
        dataframe['kama_entry_threshold'] = dataframe['kama_threshold'] * kama_entry_coeff
        dataframe['kama_exit_threshold'] = dataframe['kama_threshold'] * kamma_exit_coeff

        # Entry & Exit
        dataframe['kama_entry_pos'] = np.where(
            (dataframe['kama_delta'] > 0) &
            (dataframe['kama_delta'] > dataframe['kama_entry_threshold']), 1, 0)

        dataframe['kama_exit_pos'] = np.where(
            (dataframe['kama_delta'] < 0) &
            (dataframe['kama_delta'] < dataframe['kama_exit_threshold']), 1, 0)

        # Merge informative pairs
        # Note: `.copy` needed to avoid `date` column being removed during merge which means column
        # not available for joining for subsequent pairs
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_btc_5m.copy(), self.timeframe,
            '5m', ffill=True, date_column='date_btc')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_btc_1h.copy(), self.timeframe,
            '1h', ffill=True, date_column='date_btc')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_glassnode_1d.copy(), self.timeframe,
            '1d', ffill=True, date_column='date')
        dataframe = merge_informative_pair(
            dataframe, dataframe_low_res, self.timeframe,
            '1h', ffill=True, date_column='date')

        # Add reference to pair so ML model can generate feature for prediction
        dataframe['pair_copy'] = metadata['pair']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy trend indicator when buy_trigger crosses from 0 to 1
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with buy column
        """

        dataframe.loc[
            (qtpylib.crossed_above(dataframe['kama_entry_pos'], 0.5))
            & (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']] = (1, 'enter_long_trigger')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        """dataframe.loc[
            (qtpylib.crossed_below(dataframe['kama_exit_pos'], 0.5))
            & (dataframe['volume'] > 0),
            ['exit_long', 'enter_tag']] = (1, 'exit_long_trigger')"""

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time, current_rate: float,
                    current_profit: float, **kwargs):
        """Custom exit needed to sell all trades open after X minutes & to label that forced
        sale as such. minimal_roi was not suitable for this because it would not allow
        labeling of the forced sell reason (ie. vertical barrier vs upper barrier hit).
            ----------
            pair : Pair that's currently analyzed
            trade : trade object
            current_time : datetime object, containing the current datetime
            current_rate : Current price, calculated based on pricing settings in ask_strategy
            current_profit : Current profit (as ratio), calculated based on current_rate
            return : custom sell_reason
        """

        # Sell any positions at a loss if they are held for more than X seconds.
        vertical_barrier_seconds = 3 * 60 * 60
        if (current_time - trade.open_date_utc).seconds > vertical_barrier_seconds:
            return 'vertical_barrier_force_sell'

        # Triple Barrier Method: Upper and lower barriers based on EMA Daily Volatility
        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Upper / lower barrier multiplier
        # Does NOT need to be symmetric
        PT_MULTIPLIER = 1.03
        SL_MULTIPLIER = 0.95

        # Look up original candle on the trade date
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        # trade_candle may be empty for trades that just opened as it is still incomplete.
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()

            # Profit taking upper barrier sell trigger
            if current_rate > trade_candle['open'] * PT_MULTIPLIER:
                return 'upper_barrier_pt_sell'

            # Stop loss lower barrier sell trigger
            elif current_rate < trade_candle['open'] * SL_MULTIPLIER:
                return 'lower_barrier_sl_sell'

        return None
