import subprocess
import logging
import os
import shlex
from importlib.resources import files
from argparse import ArgumentParser

import stormworkflow
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


_logger = logging.getLogger(__file__)

def main():

    parser = ArgumentParser()
    parser.add_argument('--configuration', '-c', type=str, required=False)
    args = parser.parse_args()

    scripts = files('stormworkflow.scripts')
    slurm = files('stormworkflow.slurm')
    refs = files('stormworkflow.refs')

    infile = args.configuration
    if infile is None:
        _logger.warn('No input configuration provided, using reference file!')
        infile = refs.joinpath('input.yaml')

    with open(infile, 'r') as yfile:
        conf = yaml.load(yfile, Loader=Loader)

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
