##
# Brief:    Functions to periodically fetch external signals for freqtrade strategy.
#           Store this data locally and provide functions to fetch it within
#           freqtrade when needed
# Author:   markdregan@gmail.com

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_yfinance_data(tickers: list, file_name: str, **kwargs):
    # Get from Yahoo finance
    df = yf.download(tickers, period='5y', group_by='ticker', **kwargs)

    # Move ticker index from column to row
    df = df.stack(level=0).reorder_levels(order=[1, 0]).sort_index()

    # Rename index
    df.index.rename(names=['ticker', 'date'], inplace=True)

    # Rename columns (to match freqtrade lowercase)
    df = df.rename(columns={'Close': 'close', 'High': 'high',
                            'Low': 'low', 'Open': 'open', 'Volume': 'volume'})

    # Drop unnecessary columns
    df = df.drop(columns=['Adj Close'])

    # Save file locally
    save_data_locally(df, file_name)


def save_data_locally(df: pd.DataFrame, file_name: str):
    path = 'user_data/data/external_informative_data/'
    filepath = Path(path, file_name)

    # Save to csv locally
    df.to_csv(path_or_buf=filepath)


def load_local_data(file_name: str):
    path = 'user_data/data/external_informative_data/'
    filepath = Path(path, file_name)
    dict_df = {}

    df = pd.read_csv(filepath_or_buffer=filepath, index_col=['ticker'])
    df['date'] = pd.to_datetime(df['date'], utc=True)

    # Break into multiple dataframes and return in dict
    for g, df in df.groupby('ticker'):
        dict_df[g] = df.reset_index(level='ticker', drop=True)

    return dict_df
