# Copy of Vulcan Strategy with MetaModel added
# Author: markdregan@gmail.com

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from ta import add_all_ta_features

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta

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


class VulcanExplore(IStrategy):

    INTERFACE_VERSION = 3

    stoploss = -1
    minimal_roi = {"0": 3.0}

    # Turn on position adjustment
    position_adjustment_enable = False

    plot_config = master_plot_config

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:

        # Apply TA indicators to the pair dataframe
        dataframe = add_all_ta_features(dataframe,
                                        open="open",
                                        high="high",
                                        low="low",
                                        close="close",
                                        volume="volume",
                                        fillna=True)

        # Indicators from original strategy
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

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame,
                             metadata: dict) -> DataFrame:
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

    def populate_exit_trend(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Disable populate sell
            ----------
            dataframe: DataFrame populated with indicators
            metadata: Additional information, like the currently traded pair
            return: DataFrame with sell column
        """

        dataframe.loc[
            (qtpylib.crossed_above(dataframe['sell_trigger'], 0.5))
            & (dataframe['volume'] > 0),
            ['exit_long', 'exit_tag']] = (1, 'main_sell_trigger')

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        try:
            # Bet a portion of capital remaining
            available_stake = self.wallets.get_available_stake_amount()  # type: ignore
            stake_amount = available_stake / self.config['max_open_trades']
            print(stake_amount)
            return stake_amount
        except Exception as exception:
            print(exception)
            return None
