#!/bin/bash
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=36

if [ $INTEL_SOLVER == 0 ]; then module load $MODULES; fi

export MV2_ENABLE_AFFINITY=0
ulimit -s unlimited

set -ex

pushd ${SCHISM_DIR}
mkdir -p outputs

if [ $INTEL_SOLVER == 1 ]; then
    export SINGULARITYENV_FI_PROVIDER=tcp
    export SINGULARITY_SHELL=/bin/bash
    srun --mpi=pmi2 singularity exec ${SINGULARITY_BINDFLAGS} ${IMG} \
        ${SCHISM_EXEC} 4
else
    mpirun -np 36 singularity exec ${SINGULARITY_BINDFLAGS} ${IMG} \
        ${SCHISM_EXEC} 4
fi

if [ $? -eq 0 ]; then
    echo "Combining outputs..."
    date
    pushd outputs
    if ls hotstart* >/dev/null 2>&1; then 
        times=$(ls hotstart_* | grep -o "hotstart[0-9_]\+" | awk 'BEGIN {FS = "_"}; {print $3}'  | sort -h | uniq )
        for i in $times; do
           singularity exec ${SINGULARITY_BINDFLAGS} ${IMG} \
               combine_hotstart7 --iteration $i
        done
    fi
    popd

    singularity exec ${SINGULARITY_BINDFLAGS} ${IMG} \
        expect -f /scripts/combine_gr3.exp maxelev 1
    singularity exec ${SINGULARITY_BINDFLAGS} ${IMG} \
        expect -f /scripts/combine_gr3.exp maxdahv 3
    mv maxdahv.gr3 maxelev.gr3 -t outputs
fi


echo "Done"
date
