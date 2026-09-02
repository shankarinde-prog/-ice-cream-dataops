from datetime import datetime, timedelta, timezone
from itertools import islice
from timeit import default_timer

from cognite.client import CogniteClient
from cognite.client.config import global_config
from cognite.client.data_classes import ExtractionPipelineRun
from cognite.client.data_classes.data_modeling import NodeId
from cognite.client.data_classes.data_modeling.cdm.v1 import (
    CogniteAsset,
    CogniteTimeSeries,
)
from ice_cream_factory_api import IceCreamFactoryAPI

global_config.disable_pypi_version_check = True


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def report_ext_pipe(client: CogniteClient, status: str, message: str = None):
    ext_pipe_run = ExtractionPipelineRun(
        extpipe_external_id="ep_icapi_datapoints",
        status=status,
        message=message,
    )
    client.extraction_pipelines.runs.create(run=ext_pipe_run)


def handle(client: CogniteClient = None, data=None):
    report_ext_pipe(client, "seen")

    sites = None
    backfill = None
    hours = None
    max_hours = 336

    if data:
        sites = data.get("sites")
        backfill = data.get("backfill")
        hours = data.get("hours")

        if hours and hours > max_hours:
            print(
                f"{hours} > {max_hours}! The Ice Cream API can't serve more than "
                f"{max_hours} hours of datapoints, setting hours to {max_hours}"
            )
            hours = max_hours

    all_sites = [
        "Houston",
        "Oslo",
        "Kuala_Lumpur",
        "Hannover",
        "Nuremberg",
        "Marseille",
        "Sao_Paulo",
        "Chicago",
        "Rotterdam",
        "London",
    ]

    sites = sites or all_sites
    backfill = True if backfill is None else backfill
    hours = hours or max_hours

    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    increment = int(timedelta(hours=hours).total_seconds() * 1000)

    ice_cream_api = IceCreamFactoryAPI(
        base_url="https://ice-cream-factory.inso-internal.cognite.ai"
    )

    # FIX 1: Fetch ALL time series ONCE before the site loop
    # to avoid hanging by calling list() 10 times with limit=None
    print("Fetching all time series from icapi_dm_space...")
    all_ts = client.data_modeling.instances.list(
        instance_type=CogniteTimeSeries,
        space="icapi_dm_space",
        limit=None,
    )
    print(f"Found {len(all_ts)} total time series in icapi_dm_space")

    try:
        for site in sites:
            print(f"Getting Data Points for {site}")
            big_start = default_timer()
            this_site = site.lower()

            # Check root asset exists for this site
            sub_tree_root = client.data_modeling.instances.retrieve_nodes(
                NodeId("icapi_dm_space", this_site), node_cls=CogniteAsset
            )
            if not sub_tree_root:
                print(
                    f"----No CogniteAssets in CDF for {site}!----\n"
                    f"    Run the 'Create Cognite Asset Hierarchy' transformation!"
                )
                continue

            # FIX 2: Filter from the already-fetched list using 2-letter site prefix
            # Workaround for path=[] (path materializer not running in this project)
            time_series = [
                item for item in all_ts
                if this_site[:2].upper() in item.external_id.upper()
                and any(
                    substring in item.external_id
                    for substring in ["planned_status", "good"]
                )
            ]

            if not time_series:
                print(
                    f"  No TimeSeries found for {site}, "
                    f"took {default_timer() - big_start:.2f} seconds"
                )
                continue

            print(f"  Found {len(time_series)} time series for {site}")

            ts_instance_ids = [
                NodeId(space="icapi_dm_space", external_id=ts.external_id)
                for ts in time_series
            ]

            latest_dps = {}
            if not backfill:
                dps_latest_res = client.time_series.data.retrieve_latest(
                    instance_id=ts_instance_ids, ignore_unknown_ids=True
                )
                if dps_latest_res:
                    latest_dps = {
                        dp.instance_id.external_id: dp.timestamp
                        for dp in dps_latest_res
                        if hasattr(dp, "timestamp") and dp.timestamp
                    }

            to_insert = []
            for ts in time_series:
                latest = (
                    latest_dps.get(ts.external_id) if not backfill else None
                )
                start = latest if latest else now - increment
                end = now

                # ice_cream_api.get_datapoints already returns:
                # [{"instance_id": "EXT_ID_STRING", "datapoints": [{"timestamp": ms, "value": v}, ...]}, ...]
                dps_list = ice_cream_api.get_datapoints(
                    timeseries_ext_id=ts.external_id, start=start, end=end
                )

                # FIX 3: Convert string instance_id → NodeId required by CDF SDK
                for item in dps_list:
                    item["instance_id"] = NodeId(
                        space="icapi_dm_space",
                        external_id=item["instance_id"],
                    )

                to_insert.extend(dps_list)

                if len(to_insert) >= 20:
                    client.time_series.data.insert_multiple(datapoints=to_insert)
                    to_insert = []

            if to_insert:
                client.time_series.data.insert_multiple(datapoints=to_insert)

            print(
                f"  {hours}h of data points for {site} "
                f"took {default_timer() - big_start:.2f} seconds"
            )

        report_ext_pipe(client, "success")

    except Exception as e:
        # FIX 4: "fail" is not valid — CDF API only accepts "failure", "success", "seen"
        report_ext_pipe(client, "failure", str(e))
        raise e