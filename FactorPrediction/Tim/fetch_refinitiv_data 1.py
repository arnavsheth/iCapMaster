import json
import pandas as pd
from httpx import Client, Timeout
import numpy as np


class RefinitivClient:
    """
    Class for interacting with Global Data Master (Fund Master) API
    """

    def __init__(self):
        self._client = None

    @property
    def client(self) -> Client:
        """HTTP request client"""
        if self._client:
            return self._client

        base_url = "https://selectapi.datascope.refinitiv.com/RestApi/v1/Extractions/ExtractWithNotes"
        auth = "Token _ypBZmic2uj4c5gaxUDGdiRApxtbAPak1qB-m18TjqHp_oYAHOPHRyki5KO9pChrXFi-TgL8oKlxPIcW_OM-cGjjI0c_fLsyZP" \
               "FbCN8jaG95CItwAmtWI5q1beZSBJfKCLpSZLGZDBSu4isQggDrrCoM4bm6ZVtxhGtmPNxKQwqwUXzZhKw4xIEKw74-DmzfdN3ugiA34b" \
               "lOgflROSuo-9BA5rVlKt3qexu3bY0GSJEI0yKfSPHu6KftLXsNU0Z9JIb3b9eXxjbnvSrvmF-mOGEkjYg1IY82df3uh_KYKzrc"

        # Create headers
        headers = {
            "Authorization": auth,
            'Content-Type': "application/json; charset=utf-8",
            'Connection': "keep-alive",
        }
        # Initialize client
        self._client = Client(base_url=base_url, headers=headers, timeout=Timeout(timeout=20.0))
        return self._client


REFINITIV_CLIENT = RefinitivClient()


def query_refinitiv(asset_id):

    payload = {
        "ExtractionRequest": {
            "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.PriceHistoryExtractionRequest",
            "ContentFieldNames": [
                "RIC",
                "Trade Date",
                "Universal Close Price"
            ],
            "IdentifierList": {
                "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.InstrumentIdentifierList",
                "InstrumentIdentifiers": [
                    {
                        "Identifier": f"{asset_id}",
                        "IdentifierType": "Ric"
                    }
                ],
                "ValidationOptions": {"AllowHistoricalInstruments": False, "AllowOpenAccessInstruments": True},
                "UseUserPreferencesForValidationOptions": False
            },
            "Condition": {
                "AdjustedPrices": True,
                "QueryStartDate": "1980-01-24T00:00:00.000Z",
                "QueryEndDate": "2024-02-22T00:00:00.000Z"
            }
        }
    }

    payload_json = json.dumps(payload).encode('utf-8')
    results = REFINITIV_CLIENT.client.post(
        url="https://selectapi.datascope.refinitiv.com/RestApi/v1/Extractions/ExtractWithNotes",
        data=payload_json,
    )

    data = results.json()['Contents']
    data = {x['Trade Date']: x['Universal Close Price'] for x in data}
    data = pd.DataFrame.from_dict(data, orient='index', columns=[asset_id])
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    # convert index levels to monthly returns
    data['daily_return'] = data[asset_id].pct_change()
    data = data['daily_return'].resample('M').agg(lambda x: (x + 1).prod() - 1) * 100.
    data = data.rename(asset_id).iloc[1:]

    return data


def query_performance():

    ids = ['.RUA']

    results_map = []
    for asset_id in ids:
        data = query_refinitiv(asset_id)
        results_map.append(data)

    results_map = pd.concat(results_map, axis=1)
    results_map.to_csv('/Users/timothy.copeland/Desktop/data.csv')


if __name__ == '__main__':
    query_performance()
