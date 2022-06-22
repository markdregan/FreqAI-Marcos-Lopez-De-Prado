# KAMA Primary Strategy
# Tradingview: https://www.tradingview.com/chart/WeEVLg4V/
# Author: markdregan@gmail.com

from collections import defaultdict
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from user_data.litmus import indicator_helpers
from user_data.litmus.glassnode import download_data

import freqtrade.vendor.qtpylib.indicators as qtpylib
import gc
import numpy as np
import ta.momentum

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

    def bot_start(self, **kwargs) -> None:
        """Instantiate things before bot starts"""

        # Glassnode class for fetching data from glassnode and sqlite db
        self.gn = download_data.GlassnodeData(api_key='22HCck9cUjuvUTrEvfc50rgcL7v',
                                              directory='user_data/data/glassnode/')

        # List of glassnode features that are shared across collection of tokens
        self.gn_f = ["gn_10m__v1_metrics_addresses_active_count",
                     "gn_10m__v1_metrics_addresses_receiving_count",
                     "gn_10m__v1_metrics_indicators_nvt",
                     "gn_10m__v1_metrics_indicators_nvts",
                     "gn_10m__v1_metrics_addresses_count",
                     "gn_10m__v1_metrics_indicators_velocity",
                     "gn_10m__v1_metrics_market_marketcap_usd",
                     "gn_10m__v1_metrics_market_price_drawdown_relative",
                     "gn_10m__v1_metrics_market_price_usd",
                     "gn_10m__v1_metrics_market_price_usd_close",
                     "gn_10m__v1_metrics_supply_current",
                     "gn_10m__v1_metrics_addresses_sending_count",
                     "gn_10m__v1_metrics_addresses_new_non_zero_count",
                     "gn_10m__v1_metrics_transactions_transfers_volume_mean",
                     "gn_10m__v1_metrics_transactions_transfers_volume_median",
                     "gn_10m__v1_metrics_transactions_transfers_volume_sum",
                     "gn_10m__v1_metrics_transactions_transfers_volume_between_exchanges_sum",
                     "gn_10m__v1_metrics_distribution_balance_exchanges",
                     "gn_10m__v1_metrics_transactions_transfers_to_exchanges_count",
                     "gn_10m__v1_metrics_transactions_transfers_from_exchanges_count",
                     "gn_10m__v1_metrics_transactions_transfers_between_exchanges_count",
                     "gn_10m__v1_metrics_transactions_transfers_volume_exchanges_net",
                     "gn_10m__v1_metrics_transactions_transfers_volume_from_exchanges_mean",
                     "gn_10m__v1_metrics_transactions_transfers_volume_from_exchanges_sum",
                     "gn_10m__v1_metrics_transactions_transfers_volume_to_exchanges_mean",
                     "gn_10m__v1_metrics_transactions_transfers_volume_to_exchanges_sum",
                     "gn_10m__v1_metrics_distribution_balance_exchanges_relative",
                     "gn_10m__v1_metrics_transactions_transfers_volume_within_exchanges_sum",
                     "gn_10m__v1_metrics_addresses_sending_to_exchanges_count",
                     "gn_10m__v1_metrics_addresses_receiving_from_exchanges_count",
                     "gn_10m__v1_metrics_transactions_transfers_count",
                     "gn_10m__v1_metrics_transactions_transfers_rate"]

    def bot_loop_start(self, **kwargs) -> None:

        # Informative: BTC 5m
        self.dataframe_btc_5m = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe='5m')
        self.dataframe_btc_5m = indicator_helpers.add_all_ta_informative(
            self.dataframe_btc_5m, suffix='_btc')

        # Informative: BTC 1h
        self.dataframe_btc_1h = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe='1h')
        self.dataframe_btc_1h = indicator_helpers.add_all_ta_informative(
            self.dataframe_btc_1h, suffix='_btc')

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:

        # Apply TA indicators to the pair being traded
        dataframe = indicator_helpers.add_all_ta_informative(
            dataframe, suffix='')

        # Get glassnode signals & derive TA indicators
        token = metadata['pair'].split('/')[0]
        gn_data = defaultdict(dict)  # type: ignore
        for i, f in enumerate(self.gn_f):
            SUFFIX = '_ppo_' + str(i)
            gn_data[f]['df'] = self.gn.query_metric(table_name=f, token=token,
                                                    date_from='2021-06-01', date_to='2022-07-01',
                                                    cols_to_drop=['token', 'update_timestamp'])
            gn_data[f]['ta_df'] = indicator_helpers.add_single_ta_informative(
                gn_data[f]['df'], ta.momentum.ppo, suffix=SUFFIX, col=f)
            gn_data[f]['date_key'] = 'date' + SUFFIX

        # Add informative signals of pair at lower time resolution
        low_res_tf = '1h'
        dataframe_low_res = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=low_res_tf)
        dataframe_low_res = indicator_helpers.add_all_ta_informative(
            dataframe_low_res, suffix='')

        # KAMA
        kama_window = 14
        dataframe['kama'] = ta.momentum.kama(
            dataframe['close'], window=kama_window, pow1=2, pow2=20)
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
        # Note: For dataframes fetched in `bot_loop_start`, `.copy` is needed to avoid `date`
        # column being removed during merge which results in column
        # not available for joining for subsequent pairs

        # BTC price signals
        dataframe = merge_informative_pair(
            dataframe=dataframe, informative=self.dataframe_btc_5m.copy(), timeframe=self.timeframe,
            timeframe_inf='5m', ffill=True, date_column='date_btc')
        dataframe = merge_informative_pair(
            dataframe=dataframe, informative=self.dataframe_btc_1h.copy(), timeframe=self.timeframe,
            timeframe_inf='1h', ffill=True, date_column='date_btc')

        # Lower res pair price signals
        dataframe = merge_informative_pair(
            dataframe=dataframe, informative=dataframe_low_res, timeframe=self.timeframe,
            timeframe_inf='1h', ffill=True, date_column='date')

        # Merge all glassnode signals
        for f in gn_data.keys():
            dataframe = merge_informative_pair(
                dataframe=dataframe, informative=gn_data[f]['ta_df'], timeframe=self.timeframe,
                timeframe_inf='10m', ffill=True, date_column=gn_data[f]['date_key'])

        # Free up memory by deleting big objects and triggering garbage collection
        del gn_data
        del dataframe_low_res
        gc.collect()

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
        vertical_barrier_seconds = 4 * 60 * 60
        if (current_time - trade.open_date_utc).seconds > vertical_barrier_seconds:
            return 'vertical_barrier_force_sell'

        # Triple Barrier Method: Upper and lower barriers based on EMA Daily Volatility
        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Upper / lower barrier multiplier
        # Does NOT need to be symmetric
        PT_MULTIPLIER = 1.02
        SL_MULTIPLIER = 0.96

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
