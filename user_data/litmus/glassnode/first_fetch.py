import json
import time
import requests
import pandas as pd

API_KEY = '22HCck9cUjuvUTrEvfc50rgcL7v'

DIR = '/Users/mregan/Dev/litmus/user_data/litmus/glassnode/'
FILE = DIR + 'data/tokens_tmp.txt'

# Reading in the tokens whose metrics are to be fetched
# from Glassnode
top_tokens = []
with open(FILE) as file:
    top_tokens = file.readlines()
    top_tokens = [line.rstrip() for line in top_tokens]

# This dict will be stored as a json when code finishes executing. It contains
# a list of all valid metrics for any given coin. If a request errors out, the
# metric will be removed from the list of metrics associated with any given coin.
f = open(DIR + 'data/valid_requests.json', 'r')
valid_requests = json.load(f)
f.close()

# make API requests
for token_idx, token in enumerate(top_tokens):
    # The list of metrics associated with any given token
    metrics = valid_requests[token]

    # First request to establish scope
    res = requests.get("https://api.glassnode.com/v1/metrics/{}".format(metrics[0]),
                       params={'a': token, 'api_key': API_KEY})
    df = pd.read_json(res.text, convert_dates=['t'])
    df.rename({'v': metrics[0].replace('/', '_'),
               't': 'date'}, axis='columns', inplace=True)

    period_start = time.time()
    for i in range(1, len(metrics)):
        slept = False
        metric_count = len(metrics)
        if i >= metric_count:
            break
        metric = metrics[i]

        # Make API request
        beg = time.time()
        res = requests.get("https://api.glassnode.com/v1/metrics/{}".format(metric),
                           params={'a': token, 'api_key': API_KEY})
        end = time.time()

        # Convert to pandas dataframe
        if res.status_code == 200:  # If request was successful / Coin has this metric
            print(f"Successfully got {metric} metric for {token}")
            try:
                df2 = pd.read_json(res.text, convert_dates=['t'])

                # Merge column with existing df for given token
                if not df2.empty and df2.dtypes[df2.columns[1]] != dict:
                    df2.rename({'v': metric.replace('/', '_'),
                                't': 'date'}, axis='columns', inplace=True)
                    df = pd.merge(df, df2, on='date', how='left')

            except ValueError:
                valid_requests[token].remove(metric)
                print("Error thrown, skipping to next column.")
        else:
            print(f"Status {res.status_code} Could not get {metric} metric for {token}")
            i -= 1

            # Too many requests. Sleep until time window is refreshed
            if res.status_code == 429:
                slept = True
                time.sleep(60 - ((time.time() - period_start) % 59) + 1)
                period_start = time.time()

            # If invalid request, remove metric from metrics list
            else:
                valid_requests[token].remove(metric)

        if end - beg > 2 and not slept:
            print(f"{metric} is a slow request! Consider removing it from metrics")

        # Save to CSV
        if i == metric_count - 1:
            file_dir = '/Users/mregan/Dev/litmus/user_data/data/glassnode/'
            file_name = f'glassnode_{token}_1d.csv'
            df.to_csv(file_dir + file_name, index=False)

    # Update the list of valid requests. This does not need to be done as regularly but
    # performance is dictated by waiting on requests. So it is no harm to do some extra work
    with open(DIR + 'data/valid_requests.json', 'w') as f:
        json.dump(valid_requests, f, indent=4)
        f.close()
