from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from datetime import timedelta
from typing import Any, Dict

from cognite.client import CogniteClient
from cognite.client.data_classes.data_modeling import NodeId, ViewId
from cognite.client.data_classes.data_modeling.cdm.v1 import CogniteAsset, CogniteTimeSeries, CogniteTimeSeriesApply
from cognite.client.exceptions import CogniteNotFoundError

import numpy as np

from cognite.client.config import global_config
global_config.disable_pypi_version_check = True


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def get_time_series_for_site(client: CogniteClient, site: str, space: str, all_ts: list):
    this_site = site.lower()

    # Check root asset exists
    sub_tree_root = client.data_modeling.instances.retrieve_nodes(
        NodeId(space, this_site),
        node_cls=CogniteAsset
    )

    if not sub_tree_root:
        print(
            f"----No CogniteAssets in CDF for {site}!----\n"
            f"    Run the 'Create Cognite Asset Hierarchy' transformation!"
        )
        return []

    # WORKAROUND: path=[] because path materializer is not running in this project.
    # Instead of Prefix filter on path, filter all_ts by 2-letter site prefix in external_id.
    time_series = [
        item for item in all_ts
        if this_site[:2].upper() in item.external_id.upper()
    ]

    if not time_series:
        print(
            f"----No CogniteTimeSeries in CDF for {site}!----\n"
            f"    Run the 'Contextualize Timeseries and Assets' transformation!"
        )
        return []

    return time_series


def handle(client: CogniteClient, data: Dict[str, Any] = {}) -> None:
    lookback_minutes = None
    sites = None

    if data:
        lookback_minutes = timedelta(minutes=data.get("lookback_minutes", 60)).total_seconds() * 1000
        sites = data.get("sites")

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

    lookback_minutes = lookback_minutes or timedelta(minutes=60).total_seconds() * 1000
    sites = sites or all_sites

    print(f"Processing datapoints for these sites: {sites}")

    # FIX: Fetch ALL time series ONCE before the site loop to avoid repeated
    # expensive API calls inside ThreadPoolExecutor (one per site = 10x)
    source_space = "icapi_dm_space"
    print("Fetching all time series from icapi_dm_space...")
    all_ts = client.data_modeling.instances.list(
        instance_type=CogniteTimeSeries,
        space=source_space,
        limit=None,
    )
    print(f"Found {len(all_ts)} total time series in icapi_dm_space")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_site, client, lookback_minutes, site, all_ts)
            for site in sites
        ]
        for f in futures:
            f.result()


def process_site(client, lookback_minutes, site, all_ts):
    oee_space = "oee_ts_space"
    source_space = "icapi_dm_space"

    timeseries = get_time_series_for_site(client, site, source_space, all_ts)

    if not timeseries:
        print(f"  No TimeSeries found for {site}, skipping OEE calculation.")
        return

    asset_eids = list(set([item.external_id.split(sep=":")[0] for item in timeseries]))
    instance_ids = [NodeId(space=source_space, external_id=ts.external_id) for ts in timeseries]
    all_latest_dps = client.time_series.data.retrieve_latest(instance_id=instance_ids)

    # Organize latest datapoints by equipment for alignment
    assets_dps = {
        external_id: [
            latest_dp for latest_dp in all_latest_dps
            if external_id in latest_dp.instance_id.external_id
        ]
        for external_id in asset_eids
    }

    for asset, latest_dps in assets_dps.items():
        print(f"Calculating OEE for {asset}")
        count_node = f"NodeId({source_space}, {asset}:count)"
        good_node = f"NodeId({source_space}, {asset}:good)"
        status_node = f"NodeId({source_space}, {asset}:status)"
        planned_status_node = f"NodeId({source_space}, {asset}:planned_status)"

        end = min([dp.timestamp[0] for dp in latest_dps if latest_dps and dp.timestamp], default=None)

        if end:
            dps_df = client.time_series.data.retrieve_dataframe(
                instance_id=[dp.instance_id for dp in latest_dps],
                start=end - lookback_minutes,
                end=end,
                aggregates=["sum"],
                granularity="1m",
                include_aggregate_name=False,
                limit=None
            )

            # Frontfill because "planned_status" and "status" only have datapoints when value changes
            dps_df = dps_df.ffill()

            # Fill the rest with the opposite
            try:
                first_valid_value = dps_df[planned_status_node].loc[dps_df[planned_status_node].first_valid_index()]
            except Exception as e:
                print(f"Failed to find datapoints for {planned_status_node}, {e}")
                continue

            backfill_value = 1.0 if first_valid_value == 0.0 else 0.0
            dps_df[planned_status_node] = dps_df[planned_status_node].fillna(value=backfill_value)

            # Same for status
            first_valid_value = dps_df[status_node].loc[dps_df[status_node].first_valid_index()]
            backfill_value = 1.0 if first_valid_value == 0.0 else 0.0
            dps_df[status_node] = dps_df[status_node].fillna(value=backfill_value)

            count_dps = dps_df[count_node]
            good_dps = dps_df[good_node]
            status_dps = dps_df[status_node]
            planned_status_dps = dps_df[planned_status_node]

            total_items = len(count_dps)

            if (
                total_items != len(good_dps)
                or total_items != len(status_dps)
                or total_items != len(planned_status_dps)
            ):
                print(
                    f"""{asset}: Unable to retrieve datapoints for all required OEE timeseries"""
                )

            # Calculate the components of OEE
            off_spec_node = f"{asset}:off_spec"
            quality_node = f"{asset}:quality"
            performance_node = f"{asset}:performance"
            availability_node = f"{asset}:availability"
            oee_node = f"{asset}:oee"

            dps_df[off_spec_node] = count_dps - good_dps
            dps_df[quality_node] = good_dps / count_dps
            dps_df[performance_node] = (count_dps / status_dps) / (60.0 / 3.0)
            dps_df[availability_node] = status_dps / planned_status_dps
            dps_df[oee_node] = dps_df[quality_node] * dps_df[performance_node] * dps_df[availability_node]

            # Fill divide by zeros
            dps_df = dps_df.fillna(value=0.0)
            dps_df = dps_df.replace([np.inf, -np.inf], 0.0)

            # Drop input timeseries columns
            dps_df = dps_df.drop(columns=[count_node, good_node, status_node, planned_status_node])

            to_insert = [
                {
                    "instance_id": NodeId(space=oee_space, external_id=external_id),
                    "datapoints": list(zip(dps_df[external_id].index, dps_df[external_id]))
                }
                for external_id in dps_df.columns
            ]

            try:
                client.time_series.data.insert_multiple(to_insert)
            except CogniteNotFoundError as e:
                # Create missing OEE timeseries since they don't exist yet
                ts_to_create = []
                for node_id in e.not_found:
                    print(f"Creating CogniteTimeSeries {node_id}")
                    external_id = node_id["instanceId"]["externalId"]
                    name = external_id.split(":")
                    name[-1] = name[-1].replace("_", " ").title()
                    ts_to_create.append(
                        CogniteTimeSeriesApply(
                            space=oee_space,
                            external_id=external_id,
                            name=" ".join(name),
                            is_step=False,
                            time_series_type="numeric",
                        )
                    )
                client.data_modeling.instances.apply(ts_to_create)
                client.time_series.data.insert_multiple(to_insert)