from simon.api.lazy_simon_client import LazySIMONClient
import json
import pandas as pd


if __name__ == '__main__':
    SIMON_CLIENT = LazySIMONClient(env="prod", lazy=False)

    building_block_map = {
        "US Equities": "MA4B66MW5E27UAN26KV",
        "Non-US Developed Equities":  "MA4B66MW5E27UAEP4RT",
        "Emerging Markets Equities": "MA4B66MW5E27UANT8NZ",
        "US Fixed Income": "ABB9909TN",
        "Non-US Developed Fixed Income": "MA3JGECA05AFYHQR",
        "Emerging Markets Fixed Income": "MA4B66MW5E27UAEP4R5",
        "Commodity": "MA4B66MW5E27UAHKG47",
        "REITs": "MA4B66MW5E27UANT8DX",
        "Cash": "ABB9919TB",
        "Private Equity": "5867_-100",
        "Private Credit": "5868_-100",
        "Private Real Estate": "5869_-100",
        "Hedge Funds": "5865_-100"
    }

    results_map = []
    for asset in building_block_map:
        print(asset)

        payload = {
            "ids": [building_block_map[asset]]
        }
        SIMON_CLIENT.client.post("/simon/api/v1/ubertable/query", payload)
        results = SIMON_CLIENT.client.post(
            url="simon/api/v1/rainbow/performance/get-asset-return",
            data=json.dumps(payload).encode('utf-8')
        )
        results = results.json()[0]

        data = results['returnStream'][0]['data']
        data = {x['date']: x['value'] for x in data}
        data = pd.DataFrame.from_dict(data, orient='index', columns=[asset])
        data.index = pd.to_datetime(data.index)
        results_map.append(data)

    results_map = pd.concat(results_map, axis=1)
    results_map.to_csv('/Users/timothy.copeland/Desktop/data.csv')
