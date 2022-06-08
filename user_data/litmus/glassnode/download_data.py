# Data download class for Glassnode. Originally written by Hugh. Adapted by Mark.

import time
import requests
import pandas as pd
import pathlib


class GlassnodeData:
    def __init__(self):
        self.API_KEY = '22HCck9cUjuvUTrEvfc50rgcL7v'
        self.DIR = 'user_data/data/glassnode/'

    def get_metric_names(self, filename: str) -> list:
        """Load names of metrics from external file"""

        filepath = (pathlib.Path(__file__).parent / 'data' / filename).resolve()
        with open(filepath) as file:
            self.metric_names = file.readlines()
            self.metric_names = [line.rstrip() for line in self.metric_names]
        return self.metric_names

    def get_metric(self, token: str, metric: str, timeframe: str) -> pd.DataFrame:
        """Download historical data for a single token and metric."""

        res = requests.get("https://api.glassnode.com/v1/metrics/{}".format(metric),
                           params={'a': token,
                                   'i': timeframe,
                                   'api_key': self.API_KEY})
        # Check if metric request successful
        if self.check_request_status(res, token, metric):
            try:
                metric_df = pd.read_json(res.text, convert_dates=['t'])
                metric_df['t'] = pd.to_datetime(metric_df['t'], utc=True)
                metric_df.rename({'v': metric.replace('/', '_'),
                                  't': 'date'}, axis='columns', inplace=True)
                metric_df.set_index('date', inplace=True)
                return metric_df
            except ValueError:
                # valid_requests[token].remove(metric)
                print(f"Error thrown for {metric} in {token}, skipping to next column.")

    def get_metrics(self, token: str, metrics: list, timeframe: str, save: bool) -> pd.DataFrame:
        """Download data for multiple metrics for a given token."""

        temp_metrics = []
        while len(metrics) > 0:
            try:
                df = self.get_metric(token, metrics[0], timeframe)
                if df is not None:
                    temp_metrics.append(df)
                metrics.pop(0)
            except ValueError:
                print(f"Issue getting {metrics[0]} for {token}")
                metrics.pop(0)

        # Join all metrics together
        df = pd.concat(temp_metrics, axis=1)

        if save:
            filename = f'glassnode_{token}_{timeframe}.csv'
            self.save_data(df, filename)
            print(f'Data saved as {filename} in {self.DIR}')

        return df

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data in glassnode directory"""

        filepath = pathlib.Path(self.DIR, filename)
        df.to_csv(filepath)
        return self

    def check_request_status(self, request, token: str, metric: str):
        """Checks if request is successful type 200 and returns bool"""

        if request.status_code == 200:  # If request was successful / Coin has this metric
            print(f"Successfully got {metric} metric for {token}")
            return True
        elif request.status_code == 429:
            print(f"Status {request.status_code}. Pausing for 60 seconds...")
            time.sleep(60)
            return False
        else:
            print(f"Status {request.status_code} Could not get {metric} metric for {token}")
            return False
