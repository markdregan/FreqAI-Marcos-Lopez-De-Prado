# Data download class for Glassnode. Originally written by Hugh. Adapted by Mark.

import time
import requests
import pandas as pd
import pathlib

# Patching pandas json bug described here: https://stackoverflow.com/a/61733123/17582153
import json
pd.io.json._json.loads = lambda s, *a, **kw: json.loads(s)


class GlassnodeData:
    """Get data from glassnode and store to csv locally"""

    def __init__(self):
        self.API_KEY = '22HCck9cUjuvUTrEvfc50rgcL7v'
        self.DIR = pathlib.Path('user_data/data/glassnode/')

    def get_endpoints(self, metric_path: str = None, token: list[str] = None,
                      resolution: str = None):
        """Get available endpoints for Glassnode API"""

        ep = requests.get('https://api.glassnode.com/v2/metrics/endpoints',
                          params={'api_key': self.API_KEY})

        rows = []
        for i in ep.json():
            row = {}
            row['path'] = i['path']
            row['tier'] = i['tier']
            for a in i['assets']:
                row['token'] = a['symbol']
                for r in i['resolutions']:
                    row['resolution'] = r
                    rows.append(row.copy())

        self.ep_df = pd.DataFrame(rows)

        if metric_path is not None:
            self.ep_df = self.ep_df.loc[self.ep_df['path'] == metric_path]

        if token is not None:
            self.ep_df = self.ep_df.loc[self.ep_df['token'].isin(token)]

        if resolution is not None:
            self.ep_df = self.ep_df.loc[self.ep_df['resolution'] == resolution]

        return self.ep_df

    def get_metric(self, path: str, token: str, resolution: str):
        """Download historical data for a single token and metric."""

        while True:
            # Call Glassnode API and get result
            self.res = requests.get("https://api.glassnode.com{}".format(path),
                                    params={'a': token,
                                            'i': resolution,
                                            'api_key': self.API_KEY,
                                            's': 0,
                                            'u': int(time.time())})
            # Check if rate limiting kicked in, sleep if so
            if self.res.status_code == 429:
                print(f"Status {self.res.status_code}. Pausing for 60 seconds...")
                time.sleep(60)

            elif self.res.status_code == 200:
                # Get metric data and save to csv
                try:
                    metric_df = pd.read_json(self.res.text, convert_dates=['t'])
                    metric_df['t'] = pd.to_datetime(metric_df['t'], utc=True)
                    colname = f"gn_{path.replace('/', '_')}_{resolution}"
                    metric_df.rename({'v': colname,
                                      't': 'date'}, axis='columns', inplace=True)
                    metric_df.set_index('date', inplace=True)
                    if 'o' in metric_df.columns:
                        ix = metric_df.index
                        metric_df = pd.json_normalize(metric_df['o'])
                        metric_df.set_index(ix, inplace=True)
                    return metric_df

                except Exception as e:
                    print(f"Error thrown for {path} for {token}: {e}")
                    print(f"Response text: {self.res.text}")
                    return None
            else:
                print(f'Error: Request for {path} for {token}: {self.res.text}')
                return None

    def get_metrics(self, metric_path: str = None, token: list[str] = None, resolution: str = None):
        """Download data for multiple metrics for a given token."""

        metric_df = self.get_endpoints(metric_path, token, resolution)

        for ix, metric in metric_df.iterrows():

            try:
                df = self.get_metric(metric['path'], metric['token'], metric['resolution'])
                if df is not None:
                    filename = f"gn_{metric['path'].replace('/', '_')}_{metric['resolution']}.csv"
                    dirpath = pathlib.Path(self.DIR, metric['token'], metric['resolution'])
                    dirpath.mkdir(parents=True, exist_ok=True)
                    filepath = pathlib.Path(dirpath, filename)
                    df.to_csv(filepath)
                    print(f'Data saved as {filename} in {dirpath}')

                elif df is None:
                    print(f"Error: Empty results returned for {metric['path']}")

            except ValueError:
                print(f"Issue getting {metric['path']} for {metric['token']}")

        print('Process complete...')
