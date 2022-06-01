
import time
import pandas as pd
import aiohttp
import asyncio

API_KEY = '22HCck9cUjuvUTrEvfc50rgcL7v'

top_tokens = []
with open('/home/hugh/Code/Litmus/Glassnode/tokens.txt') as file:
    top_tokens = file.readlines()
    top_tokens = [line.rstrip() for line in top_tokens]

metrics = []
with open('/home/hugh/Code/Litmus/Glassnode/metrics.txt') as file:
    # with open('test.txt') as file:
    metrics = file.readlines()
    metrics = [line.rstrip() for line in metrics]


async def get_metric_async(session, token, metric):
    async with session.get("https://api.glassnode.com/v1/metrics/{}".format(metric),
                           params={'a': token, 'api_key': API_KEY}) as res:
        mytext = await res.text()
        metric = metric.replace('/', '_')
        try:
            df = pd.read_json(mytext, convert_dates=['t'])
            df.rename({'v': metric}, axis='columns', inplace=True)
            print(f"Successfully got {metric} for {token}")
            return df
        except ValueError:
            print(f"Error {res.status} thrown for {metric}, skipping to next column.")
            if res.status == 429:
                print(f"Retry after {res.headers.keys()}")
        return pd.DataFrame()


async def main():
    # make API requests
    for token in top_tokens:
        async with aiohttp.ClientSession() as session:
            beg_token = time.time()
            print("Starting requests for {}".format(token))
            tasks = []
            for metric in metrics:
                tasks.append(asyncio.ensure_future(get_metric_async(session, token, metric)))

            dfs = await asyncio.gather(*tasks)
            df = dfs[0]

            mid_token = time.time()
            print("Time to get all requests for {}: {}s".format(token, mid_token - beg_token))

            for i in range(1, len(dfs)):
                if not dfs[i].empty and type(dfs[i](dfs[i].columns[1])) != dict:
                    df = pd.merge(df, dfs[i], on='t', how='left')

            print("Time to merge all dfs for {}: {}s".format(token, time.time() - mid_token))

            print(f'Sending {token} to GBQ')
            beg = time.time()
            dest = f'Glassnode.{token}_data'
            df.to_gbq(dest, project_id='litmus-crypto', if_exists='replace')
            end = time.time()
            print(f'Time taken to send {token} to GBQ: {end-beg}s')

asyncio.run(main())
