import os
import glob
import logging
import tempfile
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path


import f90nml
import geopandas as gpd
import numpy as np
import pandas as pd
from coupledmodeldriver import Platform
from coupledmodeldriver.configure.forcings.base import TidalSource
from coupledmodeldriver.configure import (
    BestTrackForcingJSON,
    TidalForcingJSON,
    NationalWaterModelFocringJSON,
)
from coupledmodeldriver.generate import SCHISMRunConfiguration
from coupledmodeldriver.generate.schism.script import SchismEnsembleGenerationJob
from coupledmodeldriver.generate import generate_schism_configuration
from ensembleperturbation.perturbation.atcf import perturb_tracks
from pylib_essentials.schism_file import (
    read_schism_hgrid_cached,
    schism_bpfile,
    source_sink,
    TimeHistory,
)
from pylib_essentials.utility_functions import inside_polygon
from pyschism.mesh import Hgrid
from pyschism.forcing import NWM
from scipy import spatial
from shapely import get_coordinates
from stormevents import StormEvent
from stormevents.nhc.track import VortexTrack

import stormworkflow.prep.wwm
# TODO: Later find a clean way to package this module from SCHISM from
# src/Utility/Pre-Processing/STOFS-3D-Atl-shadow-VIMS/Pre_processing/Source_sink/Relocate/
#from relocate_source_feeder import (
#    relocate_sources,
#    v16_mandatory_sources_coor,
#)


REFS = Path('/refs')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _relocate_source_sink(schism_dir, region_shape):

    # Feeder info is generated during mesh generation
    feeder_info_file = str(REFS / 'feeder_heads_bases_v2.1.xy')
    old_ss_dir = str(schism_dir)
    hgrid_fname = str(schism_dir / 'hgrid.gr3')

    # read hgrid
    hgrid = read_schism_hgrid_cached(hgrid_fname, overwrite_cache=False)

    # read original source/sink
    original_ss = source_sink.from_files(source_dir=old_ss_dir)

    region = gpd.read_file(region_shape)
    region_coords = [get_coordinates(p) for p in region.explode(index_parts=True).exterior]

    # split source/sink into inside and outside region
    _, outside_ss = original_ss.clip_by_polygons(hgrid=hgrid, polygons_xy=region_coords,)

    # relocate sources
    relocated_ss = relocate_sources(
        old_ss_dir=old_ss_dir,  # based on the without feeder hgrid
        feeder_info_file=feeder_info_file,
        hgrid_fname=hgrid_fname,  # HGrid with feeder
        outdir=str(schism_dir / 'relocated_source_sink'),
        max_search_radius=2000,  # search radius (in meters)
        mandatory_sources_coor=v16_mandatory_sources_coor,
        relocate_map=None,
        region_list=region_coords,
    )

    # Copy original data
    original_ss.writer(str(schism_dir / 'original_source_sink'))

    # combine outside and relocated sources
    combined_ss = outside_ss + relocated_ss
    combined_ss.writer(str(schism_dir / 'combined_source_sink'))
    combined_ss.writer(str(schism_dir))


def _fix_nwm_issues(ensemble_dir, hires_shapefile):

    # Workaround for hydrology param bug #34

    schism_dirs = [ensemble_dir / 'spinup', *ensemble_dir.glob('runs/*')]
    for pth in schism_dirs:
        nm_list = f90nml.read(pth / 'param.nml')
        nm_list['opt']['if_source'] = 1
        nm_list.write(pth / 'param.nml', force=True)

        if hires_shapefile.exists():
            _relocate_source_sink(pth, hires_shapefile)


def _fix_hotstart_issue(ensemble_dir):
    hotstart_dirs = ensemble_dir.glob('runs/*')
    for pth in hotstart_dirs:
        nm_list = f90nml.read(pth / 'param.nml')
        nm_list['opt']['dramp'] = 0.0
        nm_list['opt']['drampbc'] = 0.0
        nm_list['opt']['dramp_ss'] = 0.0
        nm_list['opt']['drampwind'] = 0.0
        nm_list.write(pth / 'param.nml', force=True)


def main(args):

    track_path = args.track_file
    out_dir = args.output_directory
    dt_rng_path = args.date_range_file
    tpxo_dir = args.tpxo_dir
    nwm_file = args.nwm_file
    mesh_dir = args.mesh_directory
    hires_reg = args.hires_region
    use_wwm = args.use_wwm
    with_hydrology = args.with_hydrology
    pahm_model = args.pahm_model

    workdir = out_dir
    mesh_file = mesh_dir / 'mesh_w_bdry.grd'

    workdir.mkdir(exist_ok=True)

    dt_data = pd.read_csv(dt_rng_path, delimiter=',')
    date_1, date_2, date_3 = pd.to_datetime(dt_data.date_time).dt.strftime('%Y%m%d%H').values
    model_start_time = datetime.strptime(date_1, '%Y%m%d%H')
    model_end_time = datetime.strptime(date_2, '%Y%m%d%H')
    perturb_start = datetime.strptime(date_3, '%Y%m%d%H')
    spinup_time = timedelta(days=8)

    forcing_configurations = []
    forcing_configurations.append(
        TidalForcingJSON(resource=tpxo_dir / 'h_tpxo9.v1.nc', tidal_source=TidalSource.TPXO)
    )
    if with_hydrology:
        forcing_configurations.append(
            NationalWaterModelFocringJSON(
                resource=nwm_file,
                cache=True,
                source_json=workdir / 'source.json',
                sink_json=workdir / 'sink.json',
                pairing_hgrid=mesh_file,
            )
        )

    platform = Platform.LOCAL

    unperturbed = None
    # NOTE: Assuming the track file only contains a single advisory
    # track (either a OFCL or BEST)
    orig_track = VortexTrack.from_file(track_path)
    adv_uniq = orig_track.data.advisory.unique()
    if len(adv_uniq) != 1:
        raise ValueError('Track file has multiple advisory types!')

    advisory = adv_uniq.item()
    file_deck = 'a' if advisory != 'BEST' else 'b'

    # NOTE: Perturbers use min("forecast_time") to filter multiple
    # tracks. But for OFCL forecast simulation, the track file we
    # get has unique forecast time for only the segment we want to
    # perturb, the preceeding entries are 0-hour forecasts from
    # previous forecast_times
    track_to_perturb = VortexTrack.from_file(
        track_path,
        start_date=perturb_start,
        forecast_time=perturb_start if advisory != 'BEST' else None,
        end_date=model_end_time,
        file_deck=file_deck,
        advisories=[advisory],
    )
    track_to_perturb.to_file(workdir / 'track_to_perturb.dat', overwrite=True)
    perturbations = perturb_tracks(
        perturbations=args.num_perturbations,
        directory=workdir / 'track_files',
        storm=workdir / 'track_to_perturb.dat',
        variables=args.variables,
        sample_from_distribution=args.sample_from_distribution,
        sample_rule=args.sample_rule,
        quadrature=args.quadrature,
        start_date=perturb_start,
        end_date=model_end_time,
        overwrite=True,
        file_deck=file_deck,
        advisories=[advisory],
    )

    if perturb_start != model_start_time:
        perturb_idx = orig_track.data[orig_track.data.datetime == perturb_start].index.min()

        if perturb_idx > 0:
            # If only part of the track needs to be updated
            unperturbed_data = deepcopy(orig_track).data
            unperturbed_data.advisory = 'BEST'
            unperturbed_data.forecast_hours = 0
            unperturbed = VortexTrack(
                unperturbed_data,
                file_deck='b',
                advisories=['BEST'],
                end_date=orig_track.data.iloc[perturb_idx - 1].datetime,
            )

            # Read generated tracks and append to unpertubed section

            perturbed_tracks = glob.glob(str(workdir / 'track_files' / '*.22'))
            for pt in perturbed_tracks:
                # Fake BEST track here (in case it's not a real best)!
                perturbed_data = VortexTrack.from_file(pt).data
                perturbed_data.advisory = 'BEST'
                perturbed_data.forecast_hours = 0
                perturbed = VortexTrack(perturbed_data, file_deck='b', advisories=['BEST'],)
                full_track = pd.concat(
                    (unperturbed.fort_22(), perturbed.fort_22()), ignore_index=True
                )
                # Overwrites the perturbed-segment-only file
                full_track.to_csv(pt, index=False, header=False)

    # NOTE: Point to the original.22 file so that it is used for
    # spinup too instead of spinup trying to download!
    forcing_configurations.append(
        BestTrackForcingJSON(
            nhc_code=orig_track.nhc_code,
            interval_seconds=3600,
            nws=20,
            fort22_filename=workdir / 'track_files' / 'original.22',
            attributes={'model': pahm_model},
        )
    )

    run_config_kwargs = {
        'mesh_directory': mesh_dir,
        'modeled_start_time': model_start_time,
        'modeled_end_time': model_end_time,
        'modeled_timestep': timedelta(seconds=150),
        'tidal_spinup_duration': spinup_time,
        'forcings': forcing_configurations,
        'perturbations': perturbations,
        'platform': platform,
        #        'schism_executable': 'pschism_PAHM_TVD-VL'
    }

    run_configuration = SCHISMRunConfiguration(**run_config_kwargs,)
    run_configuration['schism']['hgrid_path'] = mesh_file
    run_configuration['schism']['attributes']['ncor'] = 1

    run_configuration.write_directory(
        directory=workdir, absolute=False, overwrite=False,
    )

    # Now generate the setup
    generate_schism_configuration(
        **{
            'configuration_directory': workdir,
            'output_directory': workdir,
            'relative_paths': True,
            'overwrite': True,
            'parallel': True,
        }
    )

    _fix_hotstart_issue(workdir)
    if with_hydrology:
        _fix_nwm_issues(workdir, hires_reg)
    if use_wwm:
        wwm.setup_wwm(mesh_file, workdir, ensemble=True)


def parse_arguments():
    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        '--track-file',
        help='path to the storm track file for parametric wind setup',
        type=Path,
        required=True,
    )

    argument_parser.add_argument(
        '--output-directory',
        required=True,
        type=Path,
        default=None,
        help='path to store generated configuration files',
    )
    argument_parser.add_argument(
        '--date-range-file',
        required=True,
        type=Path,
        help='path to the file containing simulation date range',
    )
    argument_parser.add_argument(
        '-n',
        '--num-perturbations',
        type=int,
        required=True,
        help='path to input mesh (`hgrid.gr3`, `manning.gr3` or `drag.gr3`)',
    )
    argument_parser.add_argument(
        '--tpxo-dir', required=True, type=Path, help='path to the TPXO dataset directory',
    )
    argument_parser.add_argument(
        '--nwm-file', type=Path, help='path to the NWM hydrofabric dataset',
    )
    argument_parser.add_argument(
        '--mesh-directory',
        type=Path,
        required=True,
        help='path to input mesh (`hgrid.gr3`, `manning.gr3` or `drag.gr3`)',
    )
    argument_parser.add_argument(
        '--hires-region',
        type=Path,
        required=True,
        help='path to high resolution polygon shapefile',
    )
    argument_parser.add_argument('--sample-from-distribution', action='store_true')
    argument_parser.add_argument('--sample-rule', type=str, default='random')
    argument_parser.add_argument('--quadrature', action='store_true')
    argument_parser.add_argument('--use-wwm', action='store_true')
    argument_parser.add_argument('--with-hydrology', action='store_true')
    argument_parser.add_argument('--pahm-model', choices=['gahm', 'symmetric'], default='gahm')
    argument_parser.add_argument('--variables', nargs='+', type=str)

    argument_parser.add_argument('name', help='name of the storm', type=str)

    argument_parser.add_argument('year', help='year of the storm', type=int)

    args = argument_parser.parse_args()

    return args


def cli():
    main(parse_arguments())

if __name__ == '__main__':
    cli()
