# KAMA Primary Strategy
# Tradingview: https://www.tradingview.com/chart/WeEVLg4V/
# Author: markdregan@gmail.com

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import ta.momentum


def plot_config():
    """Define plot config for strategy"""

    plot_config = {}
    plot_config['main_plot'] = {
        "kama": {
            "color": "purple"
        }
    }
    plot_config['subplots'] = {
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
    }

    return plot_config


def populate_indicators(dataframe) -> pd.DataFrame:
    """Features/columns needed to support strategy"""

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

    return dataframe


def populate_entry_trend(dataframe) -> pd.DataFrame:
    """Logic to define entry positions in strategy"""

    dataframe.loc[
        (qtpylib.crossed_above(dataframe['kama_entry_pos'], 0.5))
        & (dataframe['volume'] > 0),
        ['enter_long', 'enter_tag']] = (1, 'enter_long_trigger')

    return dataframe
