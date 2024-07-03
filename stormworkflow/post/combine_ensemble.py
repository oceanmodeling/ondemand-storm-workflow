from argparse import ArgumentParser
from pathlib import Path

from ensembleperturbation.client.combine_results import combine_results
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('klpc_wetonly')


def main(args):

    tracks_dir = args.tracks_dir
    ensemble_dir = args.ensemble_dir

    output = combine_results(
        model='schism',
        adcirc_like=True,
        filenames=['out2d_*.nc'], #only combine elevations. 
        output=ensemble_dir / 'analyze',
        directory=ensemble_dir,
        parallel=not args.sequential,
    )


def cli():
    parser = ArgumentParser()
    parser.add_argument('-d', '--ensemble-dir', type=Path)
    parser.add_argument('-t', '--tracks-dir', type=Path)
    parser.add_argument('-s', '--sequential', action='store_true')

    main(parser.parse_args())


if __name__ == '__main__':
    cli()
