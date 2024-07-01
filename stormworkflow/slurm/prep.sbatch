#!/bin/bash
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --nodes=1

set -ex

setup_ensemble ${PREP_KWDS} ${STORM} ${YEAR} 
