import logging
import tempfile
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pyschism.mesh import Hgrid
from pyschism.forcing import NWM


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    
    args = parse_arguments()

    out_dir = args.output_directory
    dt_rng_path = args.date_range_file
    nwm_file = args.nwm_file
    mesh_dir = args.mesh_directory
    with_hydrology = args.with_hydrology

    workdir = out_dir
    mesh_file = mesh_dir / 'mesh_w_bdry.grd'

    workdir.mkdir(exist_ok=True)

    dt_data = pd.read_csv(dt_rng_path, delimiter=',')
    date_1, date_2, _ = pd.to_datetime(dt_data.date_time).dt.strftime('%Y%m%d%H').values
    model_start_time = datetime.strptime(date_1, '%Y%m%d%H')
    model_end_time = datetime.strptime(date_2, '%Y%m%d%H')
    spinup_time = timedelta(days=2)

    # Right now the only download is for NWM, in the future there
    # might be more!

    if with_hydrology:
        # Trigger NWM data caching
        with tempfile.TemporaryDirectory() as tmpdir:
            # NOTE: The output of write is not important. Calling
            # `write` results in the relevant files being cached!
            nwm = NWM(nwm_file=nwm_file, cache=True)
            nwm.write(
                output_directory=tmpdir,
                gr3=Hgrid.open(mesh_file, crs=4326),
                start_date=model_start_time - spinup_time,
                end_date=model_end_time - model_start_time + spinup_time,
                overwrite=True,
            )
            nwm.pairings.save_json(
                sources=workdir / 'source.json', sinks=workdir / 'sink.json'
            )


def parse_arguments():
    argument_parser = ArgumentParser()

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
        '--nwm-file', type=Path, help='path to the NWM hydrofabric dataset',
    )
    argument_parser.add_argument(
        '--mesh-directory',
        type=Path,
        required=True,
        help='path to input mesh (`hgrid.gr3`, `manning.gr3` or `drag.gr3`)',
    )
    argument_parser.add_argument('--with-hydrology', action='store_true')

    args = argument_parser.parse_args()

    return args


if __name__ == '__main__':
    main()
