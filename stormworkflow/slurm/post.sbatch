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

storm_roc_curve \
    --storm_name ${storm} \
    --storm_year ${year} \
    --leadtime ${hr_prelandfall} \
    --prob_nc_path $ENSEMBLE_DIR/analyze/linear_k1_p1_n0.025/probabilities.nc \
    --obs_df_path ${NHC_OBS} \
    --save_dir $ENSEMBLE_DIR/analyze/linear_k1_p1_n0.025 
