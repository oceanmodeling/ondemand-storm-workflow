#!/bin/bash
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --nodes=1

set -ex

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} ${PREP_KWDS} ${STORM} ${YEAR} 
