#!/bin/bash
#SBATCH --parsable
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00

set -ex

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} ${STORM} ${YEAR} ${MESH_KWDS} 
