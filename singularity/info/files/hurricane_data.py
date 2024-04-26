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
from typing import Optional, List

import pandas as pd
import geopandas as gpd
from searvey.coops import COOPS_TidalDatum
from searvey.coops import COOPS_TimeZone
from searvey.coops import COOPS_Units
from shapely.geometry import box, base
from stormevents import StormEvent
from stormevents.nhc import VortexTrack


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S')



def trackstart_from_file(
    leadtime_file: Optional[pathlib.Path],
    nhc_code: str,
    leadtime: float,
) -> Optional[datetime]:
    if leadtime_file is None or not leadtime_file.is_file():
        return None

    leadtime_dict = pd.read_json(leadtime_file, orient='index')
    leadtime_table = leadtime_dict.drop(columns='leadtime').merge(
        leadtime_dict.leadtime.apply(
            lambda x: pd.Series({v: k for k, v in x.items()})
        ).apply(pd.to_datetime, format='%Y%m%d%H'),
        left_index=True,
        right_index=True
    ).set_index('ALnumber')

    if nhc_code.lower() not in leadtime_table.index:
        return None

    storm_all_times = leadtime_table.loc[[nhc_code.lower()]].dropna()
    if len(storm_all_times) > 1:
        storm_all_times = storm_all_times.iloc[0]
    if leadtime not in storm_all_times:
        return None

    return storm_all_times[leadtime].item()


def get_perturb_timestamp_in_track(
    track: VortexTrack,
    time_col: 'str',
    hr_before_landfall: datetime,
    prescribed: Optional[datetime],
    land_shapes: List[base.BaseGeometry],
) -> Optional[datetime]:
    '''
    For best track pick the best track time that is at least
    leadtime before the time besttrack is on land. But for forecast
    pick the track that has a fcst000 date which is 
    at least leadtime before the time that the track is on land.

    Note that for a single advisory forecast, there are still MULTIPLE
    tracks each with a different START DATE; while for best track
    there's a SINGLE track with a start date equal to the beginning.
    '''

    track_data = track.data

    assert len(set(track.advisories)) == 1

    perturb_start = track_data.track_start_time.iloc[0]
    if prescribed is not None:
        times = track_data[time_col].unique()
        leastdiff_idx = np.argmin(abs(times - prescribed))
        perturb_start = times[leastdiff_idx]
        return perturb_start

    for shp in land_shapes:
        tracks_onland = track_data[track_data.intersects(shp)]
        if not tracks_onland.empty:
            break
    else:
        # If track is never on input land polygons
        return perturb_start


    # Find tracks that started closest and prior to specified leadtime
    # For each track start date, pick the FIRST time it's on land
    candidates = tracks_onland.groupby('track_start_time').nth(0).reset_index()
    dt = timedelta(hours=hr_before_landfall)

    # Pick LAST track that starts BEFORE the given leadtime among
    # the candidates (start time and landfall time)
    candidates['timediff'] = candidates.datetime - candidates.track_start_time
    times_start_landfall = candidates[
        candidates['timediff'] >= dt
    ][
        ['track_start_time', 'datetime']
    ].iloc[-1]
    picked_track = track_data[
        track_data.track_start_time == times_start_landfall.track_start_time]

    # Get the chosen track's timestamp closest to specifid leadtime
    perturb_start = picked_track.loc[
        times_start_landfall.datetime - picked_track.datetime >= dt
    ].iloc[-1]

    return perturb_start


def main(args):

    name_or_code = args.name_or_code
    year = args.year
    date_out = args.date_range_outpath
    track_out = args.track_outpath
    swath_out = args.swath_outpath
    sta_dat_out = args.station_data_outpath
    sta_loc_out = args.station_location_outpath
    use_past_forecast = args.past_forecast
    hr_before_landfall = args.hours_before_landfall
    lead_times = args.lead_times
    track_dir = args.preprocessed_tracks_dir

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

    prescribed = trackstart_from_file(
        lead_times, nhc_code, hr_before_landfall
    )

    # TODO: Get user input for whether it's forecast or now!
    now = datetime.now()
    is_current_storm = (now - event.start_date < timedelta(days=30))

    df_dt = pd.DataFrame(columns=['date_time'])
    
    # All preprocessed tracks are treated as OFCL
    local_track_file = pathlib.Path()
    if track_dir is not None:
        local_track_file = track_dir / f'a{nhc_code.lower()}.dat'

    if use_past_forecast or is_current_storm:
        logger.info("Fetching a-deck track info...")

        advisory = 'OFCL'
        if not local_track_file.is_file():
            # Find and pick a single advisory based on priority
            temp_track = event.track(file_deck='a')
            adv_avail = temp_track.unfiltered_data.advisory.unique()
            adv_order = ['OFCL', 'HWRF', 'HMON', 'CARQ']
            advisory = adv_avail[0]
            for adv in adv_order:
                if adv in adv_avail:
                    advisory = adv
                    break

            # TODO: THIS IS NO LONGER RELEVANT IF WE FAKE RMWP AS OFCL!
            if advisory == "OFCL" and "CARQ" not in adv_avail:
                raise ValueError(
                    "OFCL advisory needs CARQ for fixing missing variables!"
                )

            track = VortexTrack(nhc_code, file_deck='a', advisories=[advisory])

        else:  # read from preprocessed file
            advisory = 'OFCL'

            # If a file exists, use the local file
            track_raw = pd.read_csv(local_track_file, header=None)
            assert len(track_raw[4].unique()) == 1
            track_raw[4] = 'OFCL'

            with tempfile.NamedTemporaryFile() as tmp:
                track_raw.to_csv(tmp.name, index=None)
                track = VortexTrack(
                    tmp.name, file_deck='a', advisories=[advisory]
                )


        forecast_start = None  # TODO?
        if is_current_storm:
            # Get the latest track forecast
            forecast_start = track.data.track_start_time.max()
            coops_ssh = None

        else: #if use_past_forecast:
            logger.info(
                f"Creating {advisory} track for {hr_before_landfall}"
                +" hours before landfall forecast..."
            )
            forecast_start = get_perturb_timestamp_in_track(
                track,
                'track_start_time',
                hr_before_landfall,
                prescribed,
                [shp_US, ne_low.unary_union],
            )

            logger.info("Fetching water levels for COOPS stations...")
            coops_ssh = event.coops_product_within_isotach(
                product='water_level', wind_speed=34,
                datum=COOPS_TidalDatum.NAVD,
                units=COOPS_Units.METRIC,
                time_zone=COOPS_TimeZone.GMT,
            )

        df_dt['date_time'] = (
            forecast_start - timedelta(days=2), track.end_date, forecast_start
        )

        gdf_track = track.data[track.data.track_start_time == forecast_start]
        # Prepend track from previous 0hr forecasts:
        gdf_track = pd.concat((
            track.data[
                (track.data.track_start_time < forecast_start)
                & (track.data.forecast_hours == 0)
            ],
            gdf_track
        ))

        # NOTE: Fake best track for PySCHISM AFTER perturbation
        # Fill missing name column if any
        gdf_track['name'] = storm_name
        track = VortexTrack(
            storm=gdf_track, file_deck='a', advisories=[advisory]
        )

        windswath_dict = track.wind_swaths(wind_speed=34)
        windswaths = windswath_dict[advisory]
        logger.info(f"Fetching {advisory} windswath...")
        windswath_time = min(pd.to_datetime(list(windswaths.keys())))
        windswath = windswaths[
            windswath_time.strftime("%Y%m%dT%H%M%S")
        ]

    else: # Best track

        logger.info("Fetching b-deck track info...")


        logger.info("Fetching BEST windswath...")
        track = event.track(file_deck='b')

        perturb_start = track.start_date
        if hr_before_landfall:
            perturb_start = get_perturb_timestamp_in_track(
                track,
                'datetime',
                hr_before_landfall,
                prescribed,
                [shp_US, ne_low.unary_union],
            )

        logger.info("Fetching water level measurements from COOPS stations...")
        coops_ssh = event.coops_product_within_isotach(
            product='water_level', wind_speed=34,
            datum=COOPS_TidalDatum.NAVD,
            units=COOPS_Units.METRIC,
            time_zone=COOPS_TimeZone.GMT,
        )

        df_dt['date_time'] = (
            track.start_date, track.end_date, perturb_start
        )

        # Drop duplicate rows based on isotach and time without minutes
        # (PaHM doesn't take minutes into account)
        gdf_track = track.data
        gdf_track.datetime = gdf_track.datetime.dt.floor('h')
        gdf_track = gdf_track.drop_duplicates(
            subset=['datetime', 'isotach_radius'], keep='last'
        )
        track = VortexTrack(
            storm=gdf_track, file_deck='b', advisories=['BEST']
        )

        windswath_dict = track.wind_swaths(wind_speed=34)
        windswaths = windswath_dict['BEST']
        latest_advistory_stamp = max(pd.to_datetime(list(windswaths.keys())))
        windswath = windswaths[
            latest_advistory_stamp.strftime("%Y%m%dT%H%M%S")
        ]

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

    parser.add_argument(
        "--preprocessed-tracks-dir",
        type=pathlib.Path,
        help="Existing adjusted track directory",
    )

    args = parser.parse_args()
    
    main(args)
