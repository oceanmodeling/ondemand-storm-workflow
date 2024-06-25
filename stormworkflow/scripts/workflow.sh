#!/bin/bash
set -e

export PATH=$PATH:$PATH_APPEND
export TMPDIR

# Processing...
mkdir -p $TMPDIR

# CHECK VER
### pip install --quiet --report - --dry-run --no-deps -r requirements.txt | jq -r '.install'

# CHECK BIN
# combine_hotstart7
# pschism ...
input_file=$1

function version {
    logfile=$1
    pip list | grep $2 >> $logfile
}

function add_sbatch_header {
    fnm=${2##*\/}
    awk '!found && /^#SBATCH/ { print "#SBATCH '$1'"; found=1 } 1' $2 > /tmp/$fnm
    mv /tmp/$fnm $2
}

function init {
    local run_dir=$RUN_OUT/$1
    mkdir $run_dir
    mkdir $run_dir/slurm
    mkdir $run_dir/mesh
    mkdir $run_dir/setup
    mkdir $run_dir/nhc_track
    mkdir $run_dir/coops_ssh

    for i in $L_SCRIPT_DIR/*.sbatch; do
        d=$run_dir/slurm/${i##*\/}
        cp $i $d
        if [ ! -z $hpc_partition ]; then
            add_sbatch_header "--partition=$hpc_partition" $d
        fi
        if [ ! -z $hpc_account ]; then
            add_sbatch_header "--account=$hpc_account" $d
        fi
    done

    logfile=$run_dir/versions.info
    version $logfile stormevents
    version $logfile ensembleperturbation
    version $logfile ocsmesh
    echo "SCHISM: see solver.version each outputs dir" >> $logfile

    cp $input_file $run_dir/input.yaml

    echo $run_dir
}

uuid=$(uuidgen)
tag=${storm}_${year}_${uuid}
if [ ! -z $suffix ]; then tag=${tag}_${suffix}; fi
run_dir=$(init $tag)
echo $run_dir

hurricane_data \
    --date-range-outpath $run_dir/setup/dates.csv \
    --track-outpath $run_dir/nhc_track/hurricane-track.dat \
    --swath-outpath $run_dir/windswath \
    --station-data-outpath $run_dir/coops_ssh/stations.nc \
    --station-location-outpath $run_dir/setup/stations.csv \
    $(if [ $past_forecast == 1 ]; then echo "--past-forecast"; fi) \
    --hours-before-landfall "$hr_prelandfall" \
    --lead-times "$L_LEADTIMES_DATASET" \
    --preprocessed-tracks-dir "$L_TRACK_DIR" \
    $storm $year


MESH_KWDS=""
if [ $subset_mesh == 1 ]; then
    MESH_KWDS+="subset_n_combine"
    MESH_KWDS+=" $L_MESH_HI"
    MESH_KWDS+=" $L_MESH_LO"
    MESH_KWDS+=" ${run_dir}/windswath"
    MESH_KWDS+=" --rasters $L_DEM_LO"
else
    # TODO: Get param_* values from somewhere
    MESH_KWDS+="hurricane_mesh"
    MESH_KWDS+=" --hmax $param_mesh_hmax"
    MESH_KWDS+=" --hmin-low $param_mesh_hmin_low"
    MESH_KWDS+=" --rate-low $param_mesh_rate_low"
    MESH_KWDS+=" --transition-elev $param_mesh_trans_elev"
    MESH_KWDS+=" --hmin-high $param_mesh_hmin_high"
    MESH_KWDS+=" --rate-high $param_mesh_rate_high"
    MESH_KWDS+=" --shapes-dir $L_SHP_DIR"
    MESH_KWDS+=" --windswath ${run_dir}/windswath"
    MESH_KWDS+=" --lo-dem $L_DEM_LO"
    MESH_KWDS+=" --hi-dem $L_DEM_HI"
fi
MESH_KWDS+=" --out ${run_dir}/mesh"
export MESH_KWDS
sbatch \
    --output "${run_dir}/slurm/slurm-%j.mesh.out" \
    --wait \
    --job-name=mesh_$tag \
    --export=ALL,MESH_KWDS,STORM=$storm,YEAR=$year \
    $run_dir/slurm/mesh.sbatch


echo "Download necessary data..."
# TODO: Separate pairing NWM-elem from downloading!
DOWNLOAD_KWDS=""
if [ $hydrology == 1 ]; then DOWNLOAD_KWDS+=" --with-hydrology"; fi
download_data \
    --output-directory $run_dir/setup/ensemble.dir/ \
    --mesh-directory $run_dir/mesh/ \
    --date-range-file $run_dir/setup/dates.csv \
    --nwm-file $L_NWM_DATASET \
    $DOWNLOAD_KWDS


echo "Setting up the model..."
PREP_KWDS+=" --track-file $run_dir/nhc_track/hurricane-track.dat"
PREP_KWDS+=" --output-directory $run_dir/setup/ensemble.dir/"
PREP_KWDS+=" --num-perturbations $num_perturb"
PREP_KWDS+=" --mesh-directory $run_dir/mesh/"
PREP_KWDS+=" --hires-region $run_dir/mesh/hires"
PREP_KWDS+=" --sample-from-distribution"
PREP_KWDS+=" --sample-rule $sample_rule"
PREP_KWDS+=" --date-range-file $run_dir/setup/dates.csv"
PREP_KWDS+=" --nwm-file $L_NWM_DATASET"
PREP_KWDS+=" --tpxo-dir $L_TPXO_DATASET"
if [ $use_wwm == 1 ]; then PREP_KWDS+=" --use-wwm"; fi
if [ $hydrology == 1 ]; then PREP_KWDS+=" --with-hydrology"; fi
PREP_KWDS+=" --pahm-model $pahm_model"
export PREP_KWDS
# NOTE: We need to wait because run jobs depend on perturbation dirs!
setup_id=$(sbatch \
    --output "${run_dir}/slurm/slurm-%j.setup.out" \
    --wait \
    --job-name=prep_$tag \
    --parsable \
    --export=ALL,PREP_KWDS,STORM=$storm,YEAR=$year,IMG="$L_IMG_DIR/prep.sif" \
    $run_dir/slurm/prep.sbatch \
)


echo "Launching runs"
SCHISM_SHARED_ENV=""
SCHISM_SHARED_ENV+="ALL"
SCHISM_SHARED_ENV+=",IMG=$L_IMG_DIR/solve.sif"
SCHISM_SHARED_ENV+=",MODULES=$L_SOLVE_MODULES"
spinup_id=$(sbatch \
    --nodes $hpc_solver_nnodes --ntasks $hpc_solver_ntasks \
    --parsable \
    --output "${run_dir}/slurm/slurm-%j.spinup.out" \
    --job-name=spinup_$tag \
    -d afterok:$setup_id \
    --export="$SCHISM_SHARED_ENV",SCHISM_EXEC="$spinup_exec" \
    $run_dir/slurm/schism.sbatch "$run_dir/setup/ensemble.dir/spinup"
)

joblist=""
for i in $run_dir/setup/ensemble.dir/runs/*; do
    jobid=$(
        sbatch --parsable -d afterok:$spinup_id \
        --nodes $hpc_solver_nnodes --ntasks $hpc_solver_ntasks \
        --output "${run_dir}/slurm/slurm-%j.run-$(basename $i).out" \
        --job-name="run_$(basename $i)_$tag" \
        --export="$SCHISM_SHARED_ENV",SCHISM_EXEC="$hotstart_exec" \
        $run_dir/slurm/schism.sbatch "$i"
        )
    joblist+=":$jobid"
done

# Post processing
sbatch \
    --parsable \
    --output "${run_dir}/slurm/slurm-%j.post.out" \
    --job-name=post_$tag \
    -d afterok${joblist} \
    --export=ALL,IMG="$L_IMG_DIR/prep.sif",ENSEMBLE_DIR="$run_dir/setup/ensemble.dir/" \
    $run_dir/slurm/post.sbatch
