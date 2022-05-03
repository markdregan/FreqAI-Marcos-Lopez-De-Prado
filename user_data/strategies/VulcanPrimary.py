# Copy of Vulcan Strategy with MetaModel added
# Author: markdregan@gmail.com

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
from ta import add_all_ta_features
from user_data.litmus import external_informative_data as eid

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta

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
            "buy_trigger": {
                "color": "green"
            },
            "sell_trigger": {
                "color": "red"
            },
        },
        "Grow & Shrink": {
            "growing_SMA": {
                "color": "green"
            },
            "shrinking_SMA": {
                "color": "red"
            },
        },
        "RSI": {
            "RSI": {
                "color": "green"
            },
            "RSI_SMA": {
                "color": "purple"
            },
        },
        "STOCH": {
            "slowd": {
                "color": "green"
            },
            "slowk": {
                "color": "purple"
            },
        },
    },
}


class VulcanPrimary(IStrategy):
    # Using the new freqtrade API version (V3)
    INTERFACE_VERSION = 3

    timeframe = "12h"

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

    def bot_loop_start(self, **kwargs) -> None:
        # Import external informative data
        self.ext_df_dict = eid.load_local_data(file_name='yfinance.csv')

        # Informative: BTC
        self.btc_tf = '12h'
        self.dataframe_btc_12h = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe=self.btc_tf)
        self.dataframe_btc_12h = eid.add_ta_informative(self.dataframe_btc_12h, suffix='_btc')

        # Informative: SPX (^GSPC)
        self.spx_tf = '1d'
        self.dataframe_spx_1d = self.ext_df_dict['^GSPC'].copy()
        self.dataframe_spx_1d = eid.add_ta_informative(self.dataframe_spx_1d, suffix='_spx')

        # Informative: DXY (DX-Y.NYB)
        self.dxy_tf = '1d'
        self.dataframe_dxy_1d = self.ext_df_dict['DX-Y.NYB'].copy()
        self.dataframe_dxy_1d = eid.add_ta_informative(self.dataframe_dxy_1d, suffix='_dxy')

        # Informative: VIX (^VIX)
        self.vix_tf = '1d'
        self.dataframe_vix_1d = self.ext_df_dict['^VIX'].copy()
        self.dataframe_vix_1d = eid.add_ta_informative(self.dataframe_vix_1d, suffix='_vix')

        # Informative: Gold (GC=F)
        self.gold_tf = '1d'
        self.dataframe_gold_1d = self.ext_df_dict['GC=F'].copy()
        self.dataframe_gold_1d = eid.add_ta_informative(self.dataframe_gold_1d, suffix='_gold')

        # Informative: Berkshire (BRK-A)
        self.berk_tf = '1d'
        self.dataframe_berk_1d = self.ext_df_dict['BRK-A'].copy()
        self.dataframe_berk_1d = eid.add_ta_informative(self.dataframe_berk_1d, suffix='_berk')

        # Informative: Ark (ARKK)
        self.ark_tf = '1d'
        self.dataframe_ark_1d = self.ext_df_dict['ARKK'].copy()
        self.dataframe_ark_1d = eid.add_ta_informative(self.dataframe_ark_1d, suffix='_ark')

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign timeframe to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '12h') for pair in pairs]
        # Add additional informative pairs
        informative_pairs += [
            ("BTC/USDT", "12h")
        ]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:

        # Apply TA indicators to the pair dataframe
        dataframe = add_all_ta_features(
            dataframe, open="open", high="high", low="low",
            close="close", volume="volume",  fillna=True)

        # Indicators from original vulcan strategy
        dataframe["RSI"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["RSI_SMA"] = dataframe["RSI"].rolling(window=50).mean()

        dataframe["SMA"] = ta.SMA(dataframe, timeperiod=23)
        dataframe["growing_SMA"] = (
            (dataframe["SMA"] > dataframe["SMA"].shift(1))
            & (dataframe["SMA"].shift(1) > dataframe["SMA"].shift(2))
            & (dataframe["SMA"].shift(2) > dataframe["SMA"].shift(3)))
        dataframe["shrinking_SMA"] = (
                (dataframe["SMA"] < dataframe["SMA"].shift(1))
                & (dataframe["SMA"].shift(1) < dataframe["SMA"].shift(2))
                & (dataframe["SMA"].shift(2) < dataframe["SMA"].shift(3)))

        stoch = ta.STOCH(
            dataframe,
            fastk_period=14,
            slowk_period=4,
            slowk_matype=0,
            slowd_period=6,
            slowd_matype=0,
        )
        dataframe["slowd"] = stoch["slowd"]
        dataframe["slowk"] = stoch["slowk"]

        # Note: this is no longer in strategy. Consider adding back.
        dataframe["stoch_sell_cross"] = (
            (dataframe["slowd"] > 75) &
            (dataframe["slowk"] > 75)) & (qtpylib.crossed_below(
                dataframe["slowk"], dataframe["slowd"]))

        dataframe["buy_trigger"] = np.where(((dataframe["close"] > dataframe["SMA"])
                                             & (dataframe["growing_SMA"])
                                             & (dataframe["RSI"] > dataframe["RSI_SMA"])
                                             & (dataframe["RSI"] > 50)),
                                            1, 0)

        dataframe["sell_trigger"] = np.where(((dataframe["close"] < dataframe["SMA"])
                                             & (dataframe["shrinking_SMA"])
                                             & (dataframe["RSI"] < dataframe["RSI_SMA"])
                                             & (dataframe["RSI"] < 50)),
                                             1, 0)

        # Merge informative pairs
        # Note: `.copy` needed to avoid `date` column being removed during merge which means column
        # not available for joining for subsequent pairs
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_btc_12h.copy(), self.timeframe,
            self.btc_tf, ffill=True, date_column='date_btc')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_spx_1d.copy(), self.timeframe,
            self.spx_tf, ffill=True, date_column='date_spx')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_dxy_1d.copy(), self.timeframe,
            self.dxy_tf, ffill=True, date_column='date_dxy')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_vix_1d.copy(), self.timeframe,
            self.vix_tf, ffill=True, date_column='date_vix')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_gold_1d.copy(), self.timeframe,
            self.gold_tf, ffill=True, date_column='date_gold')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_berk_1d.copy(), self.timeframe,
            self.berk_tf, ffill=True, date_column='date_berk')
        dataframe = merge_informative_pair(
            dataframe, self.dataframe_ark_1d.copy(), self.timeframe,
            self.ark_tf, ffill=True, date_column='date_ark')

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
            (qtpylib.crossed_above(dataframe['buy_trigger'], 0.5))
            & (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']] = (1, 'primary_buy_trigger')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Disable populate sell
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with sell column
        """

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time,
                    current_rate: float, current_profit: float,
                    **kwargs):
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
        if (current_time - trade.open_date_utc).days > 40:
            return 'vertical_barrier_force_sell'

        # Triple Barrier Method: Upper and lower barriers based on EMA Daily Volatility
        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Upper / lower barrier multiplier
        # Does NOT need to be symmetric
        PT_MULTIPLIER = 1.5
        SL_MULTIPLIER = 0.75

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
