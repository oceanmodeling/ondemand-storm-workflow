#!/bin/bash
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --nodes=1

set -ex

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} ${STORM} ${YEAR} ${MESH_KWDS} 
