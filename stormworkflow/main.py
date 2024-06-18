import subprocess
import logging
import os
from importlib.resources import files
from argparse import ArgumentParser

import stormworkflow


_logger = logging.getLogger(__file__)

def main():

    parser = ArgumentParser()
    parser.add_argument('stormname', type=str)
    parser.add_argument('stormyear', type=int)
    parser.add_argument('--suffix', type=str, required=False)
    args = parser.parse_args()

    scripts = files('stormworkflow.scripts')
    slurm = files('stormworkflow.slurm')

    wf = scripts.joinpath('workflow.sh')

    run_env = os.environ.copy()
    run_env['L_SCRIPT_DIR'] = slurm.joinpath('.')

    run_args = [wf, args.stormname, str(args.stormyear)]
    if args.suffix is not None:
        run_args.append(args.suffix)

    ps = subprocess.run(
        run_args,
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
