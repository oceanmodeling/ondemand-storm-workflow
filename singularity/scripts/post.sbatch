#!/bin/bash
#SBATCH --parsable
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1

set -ex

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} \
    combine_ensemble \
    --ensemble-dir $ENSEMBLE_DIR \
    --tracks-dir $ENSEMBLE_DIR/track_files

#singularity run ${SINGULARITY_BINDFLAGS} ${IMG} \
#    analyze_ensemble \
python ./../prep/files/analyze_ensemble.py \
    --ensemble-dir $ENSEMBLE_DIR \
    --tracks-dir $ENSEMBLE_DIR/track_files
