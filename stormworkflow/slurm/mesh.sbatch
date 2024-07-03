#!/bin/bash
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --nodes=1

set -ex

hurricane_mesh ${STORM} ${YEAR} ${MESH_KWDS} 
