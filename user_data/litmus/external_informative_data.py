##
# Brief:    Functions to periodically fetch external signals for freqtrade strategy.
#           Store this data locally and provide functions to fetch it within
#           freqtrade when needed
# Author:   markdregan@gmail.com

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_yfinance_data(tickers: list, **kwargs):
    for ticker in tickers:

        # Get from Yahoo finance
        df = yf.download(ticker, period='5y', **kwargs)

        # Sort & rename index
        df = df.sort_index()
        df.index.rename(name='date', inplace=True)

        # Rename columns (to match freqtrade lowercase)
        df = df.rename(columns={'Close': 'close', 'High': 'high',
                                'Low': 'low', 'Open': 'open', 'Volume': 'volume'})

        # Drop unnecessary columns
        df = df.drop(columns=['Adj Close'])

        # Save file locally
        file_name = 'yfinance' + ticker + '.csv'
        save_data_locally(df, file_name, folder='yfinance')


def save_data_locally(df: pd.DataFrame, file_name: str, folder: str):
    path = 'user_data/data/'
    filepath = Path(path, folder, file_name)

    # Save to csv locally
    df.to_csv(path_or_buf=filepath)


def load_local_data(file_name: str, folder: str):
    path = 'user_data/data/'
    filepath = Path(path, folder, file_name)

    df = pd.read_csv(filepath_or_buffer=filepath)
    df['date'] = pd.to_datetime(df['date'], utc=True)

    return df
