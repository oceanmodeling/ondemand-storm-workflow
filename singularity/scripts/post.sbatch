#!/bin/bash
#SBATCH --parsable
#SBATCH --time=05:00:00
#SBATCH --nodes=1

set -ex

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} \
    combine_ensemble \
    --ensemble-dir $ENSEMBLE_DIR \
    --tracks-dir $ENSEMBLE_DIR/track_files

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} \
    analyze_ensemble \
    --ensemble-dir $ENSEMBLE_DIR \
    --tracks-dir $ENSEMBLE_DIR/track_files
