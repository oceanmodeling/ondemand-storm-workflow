#!/bin/bash
#exec "$@"
cd io/$1

# MCA issue https://github.com/open-mpi/ompi/issues/4948
#mpirun --mca btl_vader_single_copy_mechanism none -np $SCHISM_NPROCS pschism_TVD-VL

# If SYS_PTRACE capability added for container we can use MCA

echo "Starting solver..."
date

set -ex

mkdir -p outputs
mpirun -np $SCHISM_NPROCS $2 4


echo "Combining outputs..."
date
# NOTE: Due to the scribed IO, there's no need to combine main output
#pushd outputs
#times=$(ls schout_* | grep -o "schout[0-9_]\+" | awk 'BEGIN {FS = "_"}; {print $3}'  | sort -h | uniq )
#for i in $times; do
#    combine_output11 -b $i -e $i
#done
#popd

# Combine hotstart
pushd outputs
if ls hotstart* >/dev/null 2>&1; then 
    times=$(ls hotstart_* | grep -o "hotstart[0-9_]\+" | awk 'BEGIN {FS = "_"}; {print $3}'  | sort -h | uniq )
    for i in $times; do
       combine_hotstart7 --iteration $i
    done
fi
popd

expect -f /scripts/combine_gr3.exp maxelev 1
expect -f /scripts/combine_gr3.exp maxdahv 3
mv maxdahv.gr3 maxelev.gr3 -t outputs


echo "Done"
date
