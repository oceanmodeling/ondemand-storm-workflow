#!/usr/bin/env python
import os
import pathlib
from datetime import datetime, timedelta, timezone
import logging
import argparse
import shutil
import hashlib
import fcntl
from time import time
import tempfile
from contextlib import contextmanager, ExitStack

import numpy as np
import pandas as pd
import geopandas as gpd
import f90nml
from matplotlib.transforms import Bbox

from pyschism import dates
from pyschism.enums import NWSType
from pyschism.driver import ModelConfig
from pyschism.forcing.bctides import iettype, ifltype
from pyschism.forcing.nws import GFS, HRRR, ERA5, BestTrackForcing
from pyschism.forcing.nws.nws2 import hrrr3
from pyschism.forcing.source_sink import NWM
from pyschism.mesh import Hgrid, gridgr3
from pyschism.mesh.fgrid import ManningsN
from pyschism.stations import Stations

import wwm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CDSAPI_URL = 'https://cds.climate.copernicus.eu/api/v2'
TPXO_LINK_PATH = pathlib.Path('~').expanduser() / '.local/share/tpxo'
NWM_LINK_PATH = pathlib.Path('~').expanduser() / '.local/share/pyschism/nwm'


@contextmanager
def pushd(directory):
    """Temporarily modify current directory

    Parameters
    ----------
    directory: str, pathlike
        the directory to use as cwd during this context

    Returns
    -------
    None
    """

    origin = os.getcwd()
    try:
        os.chdir(directory)
        yield

    finally:
        os.chdir(origin)


def get_main_cache_path(cache_dir, storm, year):

    return cache_dir / f'{storm.lower()}_{year}'


def get_meteo_cache_path(source, main_cache_path, bbox, start_date, end_date):

    m = hashlib.md5()
    m.update(np.round(bbox.corners(), decimals=2).tobytes())
    m.update(start_date.strftime('%Y-%m-%d:%H:%M:%S').encode('utf8'))
    m.update(end_date.strftime('%Y-%m-%d:%H:%M:%S').encode('utf8'))

    meteo_cache_path = main_cache_path / f'{source}_{m.hexdigest()}'
    return meteo_cache_path


@contextmanager
def cache_lock(cache_path):

    if not cache_path.exists():
        cache_path.mkdir(parents=True, exist_ok=True)

    with open(cache_path / '.cache.lock', 'w') as fp:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            yield

        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def from_meteo_cache(meteo_cache_path, sflux_dir):

    # TODO: Generalize
    # Redundant check
    if not meteo_cache_path.exists():
        return False

    contents = list(meteo_cache_path.iterdir())
    if not any(p.match('sflux_inputs.txt') for p in contents):
        return False

    logger.info('Creating sflux from cache...')

    # Copy files from cache dir to sflux dir
    for p in contents:
        dest = sflux_dir / p.relative_to(meteo_cache_path)
        if p.is_dir():
            shutil.copytree(p, dest)
        else:
            shutil.copy(p, dest)

    logger.info('Done copying cached sflux.')

    return True


def copy_meteo_cache(sflux_dir, meteo_cache_path):

    # TODO: Generalize
    logger.info('Copying cache files to main cache location...')
    # Copy files from sflux dir to cache dir

    # Clean meteo_cache_path if already populated?
    contents_dst = list(meteo_cache_path.iterdir())
    contents_dst = [p for p in contents_dst if p.suffix != '.lock']
    for p in contents_dst:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            os.remove(p)

    # Copy files from cache dir to sflux dir
    contents_src = list(sflux_dir.iterdir())
    for p in contents_src:
        dest = meteo_cache_path / p.relative_to(sflux_dir)
        if p.is_dir():
            shutil.copytree(p, dest)
        else:
            shutil.copy(p, dest)

    logger.info('Done copying cache files to main cache location.')


def setup_schism_model(
    mesh_path,
    domain_bbox_path,
    date_range_path,
    station_info_path,
    out_dir,
    main_cache_path,
    parametric_wind=False,
    nhc_track_file=None,
    storm_id=None,
    use_wwm=False,
):

    domain_box = gpd.read_file(domain_bbox_path)
    atm_bbox = Bbox(domain_box.to_crs('EPSG:4326').total_bounds.reshape(2, 2))

    schism_dir = out_dir
    schism_dir.mkdir(exist_ok=True, parents=True)
    logger.info('Calculating times and dates')
    dt = timedelta(seconds=150.0)

    # Use an integer for number of steps or a timedelta to approximate
    # number of steps internally based on timestep
    nspool = timedelta(minutes=20.0)

    # measurement days +7 days of simulation: 3 ramp, 2 prior
    # & 2 after the measurement dates
    dt_data = pd.read_csv(date_range_path, delimiter=',')
    date_1, date_2 = pd.to_datetime(dt_data.date_time).dt.strftime('%Y%m%d%H').values
    date_1 = datetime.strptime(date_1, '%Y%m%d%H')
    date_2 = datetime.strptime(date_2, '%Y%m%d%H')

    # If there are no observation data, it's hindcast mode
    hindcast_mode = (station_info_path).is_file()
    if hindcast_mode:
        # If in hindcast mode run for 4 days: 2 days prior to now to
        # 2 days after.
        logger.info('Setup hindcast mode')
        start_date = date_1 - timedelta(days=2)
        end_date = date_2 + timedelta(days=2)
    else:

        logger.info('Setup forecast mode')

        # If in forecast mode then date_1 == date_2, and simulation
        # will run for about 3 days: abou 1 day prior to now to 2 days
        # last meteo (HRRR) cycle after.
        #
        # Since HRRR forecasts are 48 hours on 6-hour cycles, find an end
        # date which is 48 hours after the latest cycle before now! Note
        # that the last cycle upload to either AWS or NOMADS server might
        # take MORE than 1 hour in realtime cases. Also the oldest
        # cycle on NOMADS is t00z from previous day
        last_meteo_cycle = np.datetime64(
            pd.DatetimeIndex([date_2 - timedelta(hours=2)]).floor('6H').values[0], 'h'
        ).tolist()
        oneday_before_last_cycle = last_meteo_cycle - timedelta(days=1)
        start_date = oneday_before_last_cycle.replace(hour=0)
        end_date = last_meteo_cycle + timedelta(days=2)

    rnday = end_date - start_date

    dramp = timedelta(days=1.0)

    hgrid = Hgrid.open(mesh_path, crs='epsg:4326')
    fgrid = ManningsN.linear_with_depth(
        hgrid, min_value=0.02, max_value=0.05, min_depth=-1.0, max_depth=-3.0
    )

    coops_stations = None
    stations_file = station_info_path
    if stations_file.is_file():
        st_data = np.genfromtxt(stations_file, delimiter=',')
        coops_stations = Stations(
            nspool_sta=nspool, crs='EPSG:4326', elev=True, u=True, v=True
        )
        for coord in st_data:
            coops_stations.add_station(coord[0], coord[1])

    atmospheric = None
    if parametric_wind:
        # NOTE: SCHISM supports parametric ofcl forecast as well
        if nhc_track_file is not None and nhc_track_file.is_file():
            atmospheric = BestTrackForcing.from_nhc_bdeck(nhc_bdeck=nhc_track_file)
        elif storm_id is not None:
            atmospheric = BestTrackForcing(storm=storm_id)
        else:
            ValueError('Storm track information is not provided!')
    else:
        # For hindcast ERA5 is used and for forecast
        # GFS and hrrr3.HRRR. Neither ERA5 nor the GFS and
        # hrrr3.HRRR combination are supported by nws2 mechanism
        pass

    logger.info('Creating model configuration ...')
    config = ModelConfig(
        hgrid=hgrid,
        fgrid=fgrid,
        iettype=iettype.Iettype3(database='tpxo'),
        ifltype=ifltype.Ifltype3(database='tpxo'),
        nws=atmospheric,
        source_sink=NWM(),
    )

    if config.forcings.nws and getattr(config.forcings.nws, 'sflux_2', None):
        config.forcings.nws.sflux_2.inventory.file_interval = timedelta(hours=6)

    logger.info('Creating cold start ...')
    # create reference dates
    coldstart = config.coldstart(
        stations=coops_stations,
        start_date=start_date,
        end_date=start_date + rnday,
        timestep=dt,
        dramp=dramp,
        dramp_ss=dramp,
        drampwind=dramp,
        nspool=timedelta(hours=1),
        elev=True,
        dahv=True,
    )

    logger.info('Writing to disk ...')
    if not parametric_wind:

        # In hindcast mode ERA5 is used manually: temporary solution

        sflux_dir = schism_dir / 'sflux'
        sflux_dir.mkdir(exist_ok=True, parents=True)

        # Workaround for ERA5 not being compatible with NWS2 object
        meteo_cache_kwargs = {
            'bbox': atm_bbox,
            'start_date': start_date,
            'end_date': start_date + rnday,
        }

        if hindcast_mode:
            meteo_cache_path = get_meteo_cache_path(
                'era5', main_cache_path, **meteo_cache_kwargs
            )
        else:
            meteo_cache_path = get_meteo_cache_path(
                'gfs_hrrr', main_cache_path, **meteo_cache_kwargs
            )

        with cache_lock(meteo_cache_path):
            if not from_meteo_cache(meteo_cache_path, sflux_dir):
                if hindcast_mode:
                    era5 = ERA5()
                    era5.write(
                        outdir=schism_dir / 'sflux',
                        start_date=start_date,
                        rnday=rnday.total_seconds() / timedelta(days=1).total_seconds(),
                        air=True,
                        rad=True,
                        prc=True,
                        bbox=atm_bbox,
                        overwrite=True,
                    )

                else:

                    with ExitStack() as stack:

                        # Just to make sure there are not permission
                        # issues for temporary data (e.g. HRRR tmpdir
                        # in current dir)
                        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
                        stack.enter_context(pushd(tempdir))

                        gfs = GFS()
                        gfs.write(
                            outdir=schism_dir / 'sflux',
                            level=1,
                            start_date=start_date,
                            rnday=rnday.total_seconds() / timedelta(days=1).total_seconds(),
                            air=True,
                            rad=True,
                            prc=True,
                            bbox=atm_bbox,
                            overwrite=True,
                        )

                        # If we should limit forecast to 2 days, then
                        # why not use old HRRR implementation? Because
                        # We have prior day, today and 1 day forecast (?)
                        # BUT the new implementation has issues getting
                        # 2day forecast!
                        hrrr = HRRR()
                        hrrr.write(
                            outdir=schism_dir / 'sflux',
                            level=2,
                            start_date=start_date,
                            rnday=rnday.total_seconds() / timedelta(days=1).total_seconds(),
                            air=True,
                            rad=True,
                            prc=True,
                            bbox=atm_bbox,
                            overwrite=True,
                        )

                #                        hrrr3.HRRR(
                #                            start_date=start_date,
                #                            rnday=rnday.total_seconds() / timedelta(days=1).total_seconds(),
                #                            record=2,
                #                            bbox=atm_bbox
                #                        )
                #                        for i, nc_file  in enumerate(sorted(pathlib.Path().glob('*/*.nc'))):
                #                            dst_air = schism_dir / "sflux" / f"sflux_air_2.{i:04d}.nc"
                #                            shutil.move(nc_file, dst_air)
                #                            pathlib.Path(schism_dir / "sflux" / f"sflux_prc_2.{i:04d}.nc").symlink_to(
                #                                dst_air
                #                            )
                #                            pathlib.Path(schism_dir / "sflux" / f"sflux_rad_2.{i:04d}.nc").symlink_to(
                #                                dst_air
                #                            )

                with open(schism_dir / 'sflux' / 'sflux_inputs.txt', 'w') as f:
                    f.write('&sflux_inputs\n/\n')

                copy_meteo_cache(sflux_dir, meteo_cache_path)

        windrot = gridgr3.Windrot.default(hgrid)
        windrot.write(schism_dir / 'windrot_geo2proj.gr3', overwrite=True)
        ## end of workaround

        # Workaround for bug #30
        coldstart.param.opt.wtiminc = coldstart.param.core.dt
        coldstart.param.opt.nws = NWSType.CLIMATE_AND_FORECAST.value
        ## end of workaround

    # Workaround for station bug #32
    if coops_stations is not None:
        coldstart.param.schout.nspool_sta = int(
            round(nspool.total_seconds() / coldstart.param.core.dt)
        )
    ## end of workaround

    with ExitStack() as stack:

        # Just to make sure there are not permission
        # issues for temporary data (e.g. HRRR tmpdir
        # in current dir)
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(pushd(tempdir))

        coldstart.write(schism_dir, overwrite=True)

    # Workardoun for hydrology param bug #34
    nm_list = f90nml.read(schism_dir / 'param.nml')
    nm_list['opt']['if_source'] = 1
    nm_list.write(schism_dir / 'param.nml', force=True)
    ## end of workaround

    ## Workaround to make sure outputs directory is copied from/to S3
    try:
        os.mknod(schism_dir / 'outputs' / '_')
    except FileExistsError:
        pass
    ## end of workaround

    if use_wwm:
        wwm.setup_wwm(mesh_path, schism_dir, ensemble=False)

    logger.info('Setup done')


def main(args):

    storm_name = str(args.name).lower()
    storm_year = str(args.year).lower()
    param_wind = args.parametric_wind

    mesh_path = args.mesh_file
    bbox_path = args.domain_bbox_file
    dt_rng_path = args.date_range_file
    st_loc_path = args.station_location_file
    out_dir = args.out
    nhc_track = None if args.track_file is None else args.track_file
    cache_path = get_main_cache_path(args.cache_dir, storm_name, storm_year)
    tpxo_dir = args.tpxo_dir
    nwm_dir = args.nwm_dir
    use_wwm = args.use_wwm

    if TPXO_LINK_PATH.is_dir():
        shutil.rmtree(TPXO_LINK_PATH)
    if NWM_LINK_PATH.is_dir():
        shutil.rmtree(NWM_LINK_PATH)
    os.symlink(tpxo_dir, TPXO_LINK_PATH, target_is_directory=True)
    os.symlink(nwm_dir, NWM_LINK_PATH, target_is_directory=True)

    setup_schism_model(
        mesh_path,
        bbox_path,
        dt_rng_path,
        st_loc_path,
        out_dir,
        cache_path,
        parametric_wind=param_wind,
        nhc_track_file=nhc_track,
        storm_id=f'{storm_name}{storm_year}',
        use_wwm=use_wwm,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--parametric-wind',
        '-w',
        help='flag to switch to parametric wind setup',
        action='store_true',
    )

    parser.add_argument(
        '--mesh-file', help='path to the file containing computational grid', type=pathlib.Path
    )

    parser.add_argument(
        '--domain-bbox-file',
        help='path to the file containing domain bounding box',
        type=pathlib.Path,
    )

    parser.add_argument(
        '--date-range-file',
        help='path to the file containing simulation date range',
        type=pathlib.Path,
    )

    parser.add_argument(
        '--station-location-file',
        help='path to the file containing station locations',
        type=pathlib.Path,
    )

    parser.add_argument('--cache-dir', help='path to the cache directory', type=pathlib.Path)

    parser.add_argument(
        '--track-file',
        help='path to the storm track file for parametric wind setup',
        type=pathlib.Path,
    )

    parser.add_argument(
        '--tpxo-dir', help='path to the TPXO database directory', type=pathlib.Path
    )

    parser.add_argument(
        '--nwm-dir', help='path to the NWM stream vector database directory', type=pathlib.Path
    )

    parser.add_argument(
        '--out', help='path to the setup output (solver input) directory', type=pathlib.Path
    )

    parser.add_argument('--use-wwm', action='store_true')

    parser.add_argument('name', help='name of the storm', type=str)

    parser.add_argument('year', help='year of the storm', type=int)

    args = parser.parse_args()

    main(args)
