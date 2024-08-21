import subprocess
import logging
import os
import shlex
import warnings
from importlib.resources import files
from argparse import ArgumentParser
from pathlib import Path

import yaml
from packaging.version import Version
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import stormworkflow

_logger = logging.getLogger(__file__)

CUR_INPUT_VER = Version('0.0.2')


def _handle_input_v0_0_1_to_v0_0_2(inout_conf):

    ver = Version(inout_conf['input_version'])

    # Only update config if specified version matches the assumed one
    if ver != Version('0.0.1'):
        return ver


    _logger.info(
        "Adding perturbation variables for persistent RMW perturbation"
    )
    inout_conf['perturb_vars'] = [
      'cross_track',
      'along_track',
      'radius_of_maximum_winds_persistent',
      'max_sustained_wind_speed',
    ]

    return Version('0.0.2')


def handle_input_version(inout_conf):

    if 'input_version' not in inout_conf:
        ver = CUR_INPUT_VER
        warnings.warn(
            f"`input_version` is NOT specified in `input.yaml`; assuming {ver}"
        )
        inout_conf['input_version'] = str(ver)
        return

    ver = Version(inout_conf['input_version'])

    if ver > CUR_INPUT_VER:
        raise ValueError(
            f"Input version not supported! Max version supported is {CUR_INPUT_VER}"
        )

    ver = _handle_input_v0_0_1_to_v0_0_2(inout_conf)

    if ver != CUR_INPUT_VER:
        raise ValueError(
            f"Could NOT update input to the latest version! Updated to {ver}"
        )

    inout_conf['input_version'] = str(ver)


def main():

    parser = ArgumentParser()
    parser.add_argument('configuration', type=Path)
    args = parser.parse_args()

    scripts = files('stormworkflow.scripts')
    slurm = files('stormworkflow.slurm')
    refs = files('stormworkflow.refs')

    infile = args.configuration
    if infile is None:
        warnings.warn(
            'No input configuration provided, using reference file!'
        )
        infile = refs.joinpath('input.yaml')

    with open(infile, 'r') as yfile:
        conf = yaml.load(yfile, Loader=Loader)

    handle_input_version(conf)
    # TODO: Write out the updated config as a yaml file

    wf = scripts.joinpath('workflow.sh')

    run_env = os.environ.copy()
    run_env['L_SCRIPT_DIR'] = slurm.joinpath('.')
    for k, v in conf.items():
        if isinstance(v, list):
            v = shlex.join(v)
        run_env[k] = str(v)

    ps = subprocess.run(
        [wf, infile],
        env=run_env,
        shell=False,
#        check=True,
        capture_output=False,
    )

    if ps.returncode != 0:
        _logger.error(ps.stderr)
    
    _logger.info(ps.stdout)


if __name__ == '__main__':

    main()
