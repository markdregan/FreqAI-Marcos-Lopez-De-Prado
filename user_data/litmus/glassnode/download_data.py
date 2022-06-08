# Data download class for Glassnode. Originally written by Hugh. Adapted by Mark.

import time
import requests
import pandas as pd
import pathlib


class GlassnodeData:
    """Get data from glassnode and store to csv locally"""

    def __init__(self, token: str, timeframe: str, metric_names_file: str = 'metrics_temp.txt'):
        self.API_KEY = '22HCck9cUjuvUTrEvfc50rgcL7v'
        self.DIR = pathlib.Path('user_data/data/glassnode/', timeframe)
        self.metric_names = self.get_metric_names(filename=metric_names_file)
        self.token = token
        self.timeframe = timeframe

    def get_metric_names(self, filename: str) -> list:
        """Load names of metrics from external file"""

        filepath = (pathlib.Path(__file__).parent / 'data' / filename).resolve()
        with open(filepath) as file:
            metric_names = file.readlines()
            metric_names = [line.rstrip() for line in metric_names]

        return metric_names

    def get_metric(self, metric: str):
        """Download historical data for a single token and metric."""

        while True:
            # Call Glassnode API and get result
            self.res = requests.get("https://api.glassnode.com/v1/metrics/{}".format(metric),
                                    params={'a': self.token,
                                            'i': self.timeframe,
                                            'api_key': self.API_KEY})
            # Check if rate limiting kicked in, sleep if so
            if self.res.status_code == 429:
                print(f"Status {self.res.status_code}. Pausing for 60 seconds...")
                time.sleep(60)

            elif self.res.status_code == 200:
                # Get metric data and save to csv
                try:
                    metric_df = pd.read_json(self.res.text, convert_dates=['t'])
                    metric_df['t'] = pd.to_datetime(metric_df['t'], utc=True)
                    metric_df.rename({'v': metric.replace('/', '_'),
                                      't': 'date'}, axis='columns', inplace=True)
                    metric_df.set_index('date', inplace=True)
                    return metric_df

                except Exception:
                    print(f'Error thrown for {metric} for {self.token}: {Exception}')
                    return None
            else:
                print(f'Request error for {metric} for {self.token}: {self.res.text}')
                return None

    def get_metrics(self):
        """Download data for multiple metrics for a given token."""

        for metric in self.metric_names:

            try:
                df = self.get_metric(metric)
                if df is not None:
                    metric_name = metric.replace('/', '_')
                    filename = f'glassnode_{self.token}_{metric_name}_{self.timeframe}.csv'
                    filepath = pathlib.Path(self.DIR, filename)
                    df.to_csv(filepath)
                    print(f'Data saved as {filename} in {self.DIR}')

                elif df is None:
                    print(f'Empty results returned for {metric}')

            except ValueError:
                print(f"Issue getting {metric} for {self.token}")

        print('Process complete...')
