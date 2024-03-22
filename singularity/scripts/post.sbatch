#!/bin/bash
#SBATCH --parsable
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=nos-surge
#SBATCH --qos=batch
#SBATCH --partition=hercules

set -ex

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} \
    combine_ensemble \
    --ensemble-dir $ENSEMBLE_DIR \
    --tracks-dir $ENSEMBLE_DIR/track_files

singularity run ${SINGULARITY_BINDFLAGS} ${IMG} \
    analyze_ensemble \
    --ensemble-dir $ENSEMBLE_DIR \
    --tracks-dir $ENSEMBLE_DIR/track_files
