"""User script to get hurricane info relevant to the workflow
This script gether information about:
    - Hurricane track
    - Hurricane windswath
    - Hurricane event dates
    - Stations info for historical hurricane
"""

import sys
import logging
import pathlib
import argparse
import tempfile
import numpy as np
from datetime import datetime, timedelta

import pandas as pd
import geopandas as gpd
from searvey.coops import COOPS_TidalDatum
from searvey.coops import COOPS_TimeZone
from searvey.coops import COOPS_Units
from shapely.geometry import box
from stormevents import StormEvent
from stormevents.nhc import VortexTrack


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S')


def main(args):

    name_or_code = args.name_or_code
    year = args.year
    date_out = args.date_range_outpath
    track_out = args.track_outpath
    swath_out = args.swath_outpath
    sta_dat_out = args.station_data_outpath
    sta_loc_out = args.station_location_outpath
    is_past_forecast = args.past_forecast
    hr_before_landfall = args.hours_before_landfall
    lead_times = args.lead_times

    if hr_before_landfall < 0:
        hr_before_landfall = 48

    ne_low = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    shp_US = ne_low[ne_low.name.isin(['United States of America', 'Puerto Rico'])].unary_union

    logger.info("Fetching hurricane info...")
    event = None
    if year == 0:
        event = StormEvent.from_nhc_code(name_or_code)
    else:
        event = StormEvent(name_or_code, year)
    nhc_code = event.nhc_code
    storm_name = event.name
    logger.info("Fetching a-deck track info...")

    prescribed = None
    if lead_times is not None and lead_times.is_file():
        leadtime_dict = pd.read_json(lead_times, orient='index')
        leadtime_table = leadtime_dict.drop(columns='leadtime').merge(
            leadtime_dict.leadtime.apply(
                lambda x: pd.Series({v: k for k, v in x.items()})
            ).apply(pd.to_datetime, format='%Y%m%d%H'),
            left_index=True,
            right_index=True
        ).set_index('ALnumber')

        if nhc_code.lower() in leadtime_table.index:
            storm_all_times = leadtime_table.loc[nhc_code.lower()].dropna()
            if len(storm_all_times.shape) > 1:
                storm_all_times = storm_all_times.iloc[0]
            if hr_before_landfall in storm_all_times:
                prescribed = storm_all_times[hr_before_landfall]

    # TODO: Get user input for whether its forecast or now!
    now = datetime.now()
    df_dt = pd.DataFrame(columns=['date_time'])
    if (is_past_forecast or (now - event.start_date < timedelta(days=30))):
        temp_track = event.track(file_deck='a')
        adv_avail = temp_track.unfiltered_data.advisory.unique()
        adv_order = ['OFCL', 'HWRF', 'HMON', 'CARQ']
        advisory = adv_avail[0]
        for adv in adv_order:
            if adv in adv_avail:
                advisory = adv
                break

        if advisory == "OFCL" and "CARQ" not in adv_avail:
            raise ValueError(
                "OFCL advisory needs CARQ for fixing missing variables!"
            )

        # NOTE: Track taken from `StormEvent` object is up to now only.
        # See GitHub issue #57 for StormEvents
        track = VortexTrack(nhc_code, file_deck='a', advisories=[advisory])


        if is_past_forecast:

            logger.info(
                f"Creating {advisory} track for {hr_before_landfall}"
                +" hours before landfall forecast..."
            )
            if prescribed is not None:
                start_times = track.data.track_start_time.unique()
                leastdiff_idx = np.argmin(abs(start_times - prescribed))
                forecast_start = start_times[leastdiff_idx]


            else:
                onland_adv_tracks = track.data[track.data.intersects(shp_US)]
                if onland_adv_tracks.empty:
                    # If it doesn't landfall on US, check with other countries
                    onland_adv_tracks = track.data[
                        track.data.intersects(ne_low.unary_union)
                    ]

                candidates = onland_adv_tracks.groupby('track_start_time').nth(0).reset_index()
                candidates['timediff'] = candidates.datetime - candidates.track_start_time
                forecast_start = candidates[
                    candidates['timediff'] >= timedelta(hours=hr_before_landfall)
                ].track_start_time.iloc[-1]

            gdf_track = track.data[track.data.track_start_time == forecast_start]
            # Append before track from previous forecasts:
            gdf_track = pd.concat((
                track.data[
                    (track.data.track_start_time < forecast_start)
                    & (track.data.forecast_hours == 0)
                ],
                gdf_track
            ))
            df_dt['date_time'] = (
                forecast_start - timedelta(days=2), track.end_date, forecast_start
            )


            logger.info("Fetching water level measurements from COOPS stations...")
            coops_ssh = event.coops_product_within_isotach(
                product='water_level', wind_speed=34,
                datum=COOPS_TidalDatum.NAVD,
                units=COOPS_Units.METRIC,
                time_zone=COOPS_TimeZone.GMT,
            )

        else:
            # Get the latest track forecast
            forecast_start = track.data.track_start_time.max()
            gdf_track = track.data[track.data.track_start_time == forecast_start]
            gdf_track = pd.concat((
                track.data[
                    (track.data.track_start_time < forecast_start)
                    & (track.data.forecast_hours == 0)
                ],
                gdf_track
            ))

            # Put both dates as now(), for pyschism to setup forecast
            df_dt['date_time'] = (
                track.start_date, track.end_date, forecast_start
            )

            coops_ssh = None

        # NOTE: Fake besttrack: Since PySCHISM supports "BEST" track
        # files for its parametric forcing, write track as "BEST" after
        # fixing the OFCL by CARQ through StormEvents
        # NOTE: Fake best track AFTER perturbation
#        gdf_track.advisory = 'BEST'
#        gdf_track.forecast_hours = 0
        gdf_track['name'] = storm_name
        track = VortexTrack(storm=gdf_track, file_deck='a', advisories=[advisory])

        windswath_dict = track.wind_swaths(wind_speed=34)
        # NOTE: Fake best track AFTER perturbation
#        windswaths = windswath_dict['BEST']  # Faked BEST
        windswaths = windswath_dict[advisory]
        logger.info(f"Fetching {advisory} windswath...")
        windswath_time = min(pd.to_datetime(list(windswaths.keys())))
        windswath = windswaths[
            windswath_time.strftime("%Y%m%dT%H%M%S")
        ]

    else:

        logger.info("Fetching b-deck track info...")


        logger.info("Fetching BEST windswath...")
        track = event.track(file_deck='b')
        # Drop duplicate rows based on isotach and time without minutes
        # (PaHM doesn't take minutes into account)
        gdf_track = track.data
        gdf_track.datetime = gdf_track.datetime.dt.floor('h')
        gdf_track = gdf_track.drop_duplicates(subset=['datetime', 'isotach_radius'], keep='last')
        track = VortexTrack(storm=gdf_track, file_deck='b', advisories=['BEST'])

        perturb_start = track.start_date
        if hr_before_landfall:
            if prescribed is not None:
                # NOTE: track_start_time is the genesis for best track
                times = track.data.datetime.unique()
                leastdiff_idx = np.argmin(abs(times - prescribed))
                perturb_start = times[leastdiff_idx]
            else:
                onland_adv_tracks = track.data[track.data.intersects(shp_US)]
                if onland_adv_tracks.empty:
                    # If it doesn't landfall on US, check with other countries
                    onland_adv_tracks = track.data[
                        track.data.intersects(ne_low.unary_union)
                    ]
                onland_date = onland_adv_tracks.datetime.iloc[0]
                perturb_start = track.data[
                    onland_date - track.data.datetime >= timedelta(hours=hr_before_landfall)
                ].datetime.iloc[-1]

        df_dt['date_time'] = (
            track.start_date, track.end_date, perturb_start
        )

        windswath_dict = track.wind_swaths(wind_speed=34)
        # NOTE: event.start_date (first advisory date) doesn't
        # necessarily match the windswath key which comes from track
        # start date for the first advisory (at least in 2021!)
        windswaths = windswath_dict['BEST']
        latest_advistory_stamp = max(pd.to_datetime(list(windswaths.keys())))
        windswath = windswaths[
            latest_advistory_stamp.strftime("%Y%m%dT%H%M%S")
        ]

        logger.info("Fetching water level measurements from COOPS stations...")
        coops_ssh = event.coops_product_within_isotach(
            product='water_level', wind_speed=34,
            datum=COOPS_TidalDatum.NAVD,
            units=COOPS_Units.METRIC,
            time_zone=COOPS_TimeZone.GMT,
        )

    logger.info("Writing relevant data to files...")
    df_dt.to_csv(date_out)
    # Remove duplicate entries for similar isotach and time
    # (e.g. Dorian19 and Ian22 best tracks)
    track.to_file(track_out)
    gs = gpd.GeoSeries(windswath)
    gdf_windswath = gpd.GeoDataFrame(
        geometry=gs, data={'RADII': len(gs) * [34]}, crs="EPSG:4326"
    )
    gdf_windswath.to_file(swath_out)
    if coops_ssh is not None and len(coops_ssh) > 0:
        coops_ssh.to_netcdf(sta_dat_out, 'w')
        coops_ssh[['x', 'y']].to_dataframe().drop(columns=['nws_id']).to_csv(
                sta_loc_out, header=False, index=False)

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name_or_code", help="name or NHC code of the storm", type=str)
    parser.add_argument(
        "year", help="year of the storm", type=int)

    parser.add_argument(
        "--date-range-outpath",
        help="output date range",
        type=pathlib.Path,
        required=True
    )

    parser.add_argument(
        "--track-outpath",
        help="output hurricane track",
        type=pathlib.Path,
        required=True
    )

    parser.add_argument(
        "--swath-outpath",
        help="output hurricane windswath",
        type=pathlib.Path,
        required=True
    )

    parser.add_argument(
        "--station-data-outpath",
        help="output station data",
        type=pathlib.Path,
        required=True
    )

    parser.add_argument(
        "--station-location-outpath",
        help="output station location",
        type=pathlib.Path,
        required=True
    )

    parser.add_argument(
        "--past-forecast",
        help="Get forecast data for a past storm",
        action='store_true',
    )

    parser.add_argument(
        "--hours-before-landfall",
        help="Get forecast data for a past storm at this many hour before landfall",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--lead-times",
        type=pathlib.Path,
        help="Helper file for prescribed lead times",
    )

    args = parser.parse_args()
    
    main(args)
