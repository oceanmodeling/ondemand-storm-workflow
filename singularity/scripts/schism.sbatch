#!/bin/bash
#SBATCH --parsable
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --account=nos-surge
#SBATCH --qos=batch
#SBATCH --partition=hercules

set -ex

pushd ${SCHISM_DIR}
mkdir -p outputs

export BIN_DIR=/work2/noaa/nos-surge/smani/bin/

module purge
module load python
module load intel-oneapi-compilers/2022.2.1 intel-oneapi-mpi/2021.7.1
module load netcdf-c/4.9.0 netcdf-fortran/4.6.0
module list

export MV2_ENABLE_AFFINITY=0
ulimit -s unlimited

srun $BIN_DIR/pschism_HERCULES_PAHM_TVD-VL 4

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
