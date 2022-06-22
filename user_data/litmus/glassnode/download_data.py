# Data download class for Glassnode. Originally written by Hugh. Adapted by Mark.

import datetime
import logging
import pandas as pd
import pangres
import pathlib
import requests
import sqlalchemy
import time

# Patching pandas json bug described here: https://stackoverflow.com/a/61733123/17582153
import json
pd.io.json._json.loads = lambda s, *a, **kw: json.loads(s)

# Setup logging config
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class GlassnodeData:
    """Get data from glassnode and store to csv locally"""

    def __init__(self, api_key: str, directory: str):
        self.API_KEY = api_key
        self.DIR = pathlib.Path(directory)

        # Initialize sqlite database
        connection_string = "sqlite:///litmus_external_signals.sqlite"
        self.db_engine = sqlalchemy.create_engine(connection_string)

    def get_endpoints(self, metric_path: list[str] = None, token: list[str] = None,
                      resolution: list[str] = None):
        """Get available endpoints (metrics) for Glassnode API"""

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
            self.ep_df = self.ep_df.loc[self.ep_df['path'].isin(metric_path)]

        if token is not None:
            self.ep_df = self.ep_df.loc[self.ep_df['token'].isin(token)]

        if resolution is not None:
            self.ep_df = self.ep_df.loc[self.ep_df['resolution'].isin(resolution)]

        logger.info("Got available endpoints from Glassnode")

        return self.ep_df

    def get_metric(self, path: str, token: str, resolution: str):
        """Download historical data for a single token and metric."""

        while True:
            # Call Glassnode API and get result
            logger.info(f"Requesting data from Glassnode for {path}, {token}, {resolution}")
            params = {'a': token,
                      'i': resolution,
                      'api_key': self.API_KEY,
                      's': 1420074000,  # 2015
                      'u': int(time.time())}
            self.res = requests.get(
                "https://api.glassnode.com{}".format(path), params)  # type: ignore

            # Check if rate limiting kicked in, sleep if so
            if self.res.status_code == 429:
                print(f"Status {self.res.status_code}. Pausing for 60 seconds...")
                time.sleep(60)

            elif self.res.status_code == 200:
                # Get metric data and save to csv
                logger.info("Successful API request 200")
                try:
                    metric_df = pd.read_json(self.res.text, convert_dates=['t'])
                    metric_df['t'] = pd.to_datetime(metric_df['t'], utc=True)
                    metric_df['token'] = token
                    # Cast to str to avoid sqlite int too big error
                    if 'v' in metric_df.columns:
                        metric_df['v'] = metric_df['v'].astype(str)
                    colname = f"gn_{resolution}_{path.replace('/', '_')}"
                    metric_df.rename({'v': colname,
                                      't': 'date'}, axis='columns', inplace=True)
                    metric_df.set_index(['token', 'date'], inplace=True)
                    if 'o' in metric_df.columns:
                        ix = metric_df.index
                        metric_df = pd.json_normalize(metric_df['o'])
                        # Cast to str to avoid sqlite int too big error
                        metric_df = metric_df.astype(str)
                        metric_df.set_index(ix, inplace=True)
                    return metric_df

                except Exception as e:
                    logger.error(f"Error thrown for {path} for {token}: {e}")
                    logger.error(f"Response text: {self.res.text}")
                    return None
            else:
                logger.warning(f'Error: Request for {path} for {token}: {self.res.text}')
                return None

    def get_metrics(self, metric_path: list[str] = None, token: list[str] = None,
                    resolution: list[str] = None):
        """Download data for multiple metrics for a given token."""

        metric_df = self.get_endpoints(metric_path, token, resolution)

        for ix, metric in metric_df.iterrows():

            try:
                df = self.get_metric(metric['path'], metric['token'], metric['resolution'])
                if df is not None:
                    metric_name = f"gn_{metric['resolution']}_{metric['path'].replace('/', '_')}"
                    df['update_timestamp'] = datetime.datetime.utcnow()
                    self.save_to_db(df, metric_name)

                elif df is None:
                    logger.warning(f"Error: Empty results returned for {metric['path']}")

            except Exception as e:
                logger.error(f"Issue getting {metric['path']} for {metric['token']}: {e}")

        logger.info('Process complete...')

    def save_to_db(self, df, table_name):
        """Save datafrmae to splite database"""

        try:
            pangres.upsert(con=self.db_engine, df=df, table_name=table_name, if_row_exists='update',
                           chunksize=1000, create_table=True)
            logger.info(f"Successfully saved {table_name} to database")
        except Exception as e:
            logger.error(f"Error saving {table_name} to database: {e}")

        return self

    def query_metric(self, table_name, token: str, date_from: str = None,
                     date_to: str = None, cols_to_drop: list = None, ffill: bool = False):
        """Execute SQL query and see results"""

        base_query = f"SELECT * FROM {table_name}"
        token_filter = f"WHERE token = '{token}'"

        if date_from is not None:
            date_from_filter = f"AND date >= '{date_from}'"
        else:
            date_from_filter = ""

        if date_to is not None:
            date_to_filter = f"AND date <= '{date_to}'"
        else:
            date_to_filter = ""

        sql_query = " ".join([base_query, token_filter, date_from_filter, date_to_filter])
        df = pd.read_sql(sql=sql_query, con=self.db_engine)
        logger.info("Executed SQL query")
        df['date'] = pd.to_datetime(df['date'], utc=True)

        # Drop cols not required
        if cols_to_drop is not None:
            df.drop(columns=cols_to_drop, inplace=True)

        cols_to_skip = ['date', 'token', 'update_timestamp']
        cols = [c for c in df.columns if c not in cols_to_skip]
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # Forward Fill to address NaNs
        if ffill is True:
            df[cols] = df[cols].ffill()

        return df
