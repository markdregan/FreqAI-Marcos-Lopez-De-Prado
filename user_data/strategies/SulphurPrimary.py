# Sulphur (acid) strategy.
# Author: markdregan@gmail.com

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from ta import add_all_ta_features
from user_data.litmus import external_informative_data as eid
from user_data.litmus import indicator_helpers

import freqtrade.vendor.qtpylib.indicators as qtpylib

# Prevent pandas complaining with future warning errors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

master_plot_config = {
    "main_plot": {
        "SMA": {
            "color": "red"
        },
    },
    "subplots": {
        "Buy & Sell": {
            "ha_is_green": {
                "color": "green"
            },
        },
    },
}


class SulphurPrimary(IStrategy):

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

        # Informative: BTC 5m
        self.dataframe_btc_5m = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe='5m')
        self.dataframe_btc_5m = eid.add_ta_informative(self.dataframe_btc_5m, suffix='_btc_5m')

        # Informative: BTC 1h
        self.dataframe_btc_1h = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe='1h')
        self.dataframe_btc_1h = eid.add_ta_informative(self.dataframe_btc_1h, suffix='_btc_1h')

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:

        # Apply TA indicators to the pair dataframe
        dataframe = add_all_ta_features(
            dataframe, open="open", high="high", low="low",
            close="close", volume="volume",  fillna=True)

        # Kama
        # TODO

        # Heiken Ashi
        ha_columns = ['ha_open', 'ha_high', 'ha_low', 'ha_close', 'ha_is_green']
        dataframe[ha_columns] = indicator_helpers.HA(
            dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])

        # Merge informative pairs
        # Note: `.copy` needed to avoid `date` column being removed during merge which means column
        # not available for joining for subsequent pairs
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_btc_5m.copy(), self.timeframe,
            '5m', ffill=True, date_column='date_btc_5m')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_btc_1h.copy(), self.timeframe,
            '1h', ffill=True, date_column='date_btc_1h')

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
            (qtpylib.crossed_above(dataframe['ha_is_green'], 0.5))
            & (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']] = (1, 'enter_long_trigger')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (qtpylib.crossed_below(dataframe['ha_is_green'], 0.5))
            & (dataframe['volume'] > 0),
            ['exit_long', 'enter_tag']] = (1, 'exit_long_trigger')

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

        # Sell any positions at a loss if they are held for more than X hours.
        if (current_time - trade.open_date_utc).seconds > 150:
            return 'vertical_barrier_force_sell'

        # Triple Barrier Method: Upper and lower barriers based on EMA Daily Volatility
        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Upper / lower barrier multiplier
        # Does NOT need to be symmetric
        PT_MULTIPLIER = 1.02
        SL_MULTIPLIER = 0.98

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
