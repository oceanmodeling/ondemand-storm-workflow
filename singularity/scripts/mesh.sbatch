#!/bin/bash
#SBATCH --parsable
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=nos-surge
#SBATCH --qos=batch
#SBATCH --partition=hercules

set -ex

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} ${STORM} ${YEAR} ${MESH_KWDS} 
