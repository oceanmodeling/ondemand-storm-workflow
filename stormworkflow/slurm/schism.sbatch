#!/bin/bash
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --time=03:00:00

set -ex

SCHISM_DIR=$1
pushd $SCHISM_DIR
mkdir -p outputs


if [ ! -z "$MODULES" ]; then
    module purge
    module load $MODULES
    module list
fi

export MV2_ENABLE_AFFINITY=0
ulimit -s unlimited

date
${SCHISM_EXEC} -v > outputs/solver.version
mpirun -np $SLURM_NTASKS ${SCHISM_EXEC} 4

if [ $? -eq 0 ]; then
    echo "Combining outputs..."
    date
    pushd outputs
    if ls hotstart* >/dev/null 2>&1; then 
        times=$(ls hotstart_* | grep -o "hotstart[0-9_]\+" | awk 'BEGIN {FS = "_"}; {print $3}'  | sort -h | uniq )
        for i in $times; do
           combine_hotstart7 --iteration $i
        done
    fi
    popd
fi


echo "Done"
date
