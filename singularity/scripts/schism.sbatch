#!/bin/bash
#SBATCH --parsable
#SBATCH --nodes=10
#SBATCH --ntasks=360
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=03:00:00

set -ex

pushd ${SCHISM_DIR}
mkdir -p outputs

export BIN_DIR=/nhc/bin/

module purge
module load intel/2022.1.2 impi/2022.1.2
module load netcdf
module list

export MV2_ENABLE_AFFINITY=0
ulimit -s unlimited

mpirun -np $SLURM_NTASKS $BIN_DIR/pschism_PAHM_TVD-VL 4

if [ $? -eq 0 ]; then
    echo "Combining outputs..."
    date
    pushd outputs
    if ls hotstart* >/dev/null 2>&1; then 
        times=$(ls hotstart_* | grep -o "hotstart[0-9_]\+" | awk 'BEGIN {FS = "_"}; {print $3}'  | sort -h | uniq )
        for i in $times; do
           $BIN_DIR/combine_hotstart7 --iteration $i
        done
    fi
    popd

#    singularity exec ${SINGULARITY_BINDFLAGS} ${IMG} \
#        expect -f /scripts/combine_gr3.exp maxelev 1
#    singularity exec ${SINGULARITY_BINDFLAGS} ${IMG} \
#        expect -f /scripts/combine_gr3.exp maxdahv 3
#    mv maxdahv.gr3 maxelev.gr3 -t outputs
fi


echo "Done"
date
