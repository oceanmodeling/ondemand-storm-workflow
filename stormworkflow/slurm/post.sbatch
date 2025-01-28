#!/bin/bash
#SBATCH --parsable
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

set -ex

combine_ensemble \
    --ensemble-dir $ENSEMBLE_DIR \
    --tracks-dir $ENSEMBLE_DIR/track_files

analyze_ensemble \
    --ensemble-dir $ENSEMBLE_DIR \
    --tracks-dir $ENSEMBLE_DIR/track_files

storm_roc_ts_rel_curves \
    --storm ${storm} \
    --year ${year} \
    --leadtime ${hr_prelandfall} \
    --obs_df_path ${NHC_OBS} \
    --ensemble-dir $ENSEMBLE_DIR 
