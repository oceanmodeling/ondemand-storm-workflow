#!/bin/bash

THIS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $THIS_SCRIPT_DIR/input.conf

TEST_OUT=/nhc/Soroosh.Mani/runs/
SINGULARITY_ROOT=$L_SCRIPT_DIR/../

init () {
    uuid=$(uuidgen)
    tag=test_${2}_${3}_${uuid}

    local run_dir=$1/$tag
    mkdir $run_dir
    echo $run_dir
}

test_hurricane_info () {
    storm=florence
    year=2018
    hr_prelandfall=48

    run_dir=$(init $TEST_OUT $storm $year)

    this_test_out=$run_dir/info_w_leadjson_preptrack_$hr_prelandfall
    mkdir $this_test_out
    python $SINGULARITY_ROOT/info/files/hurricane_data.py \
        --date-range-outpath $this_test_out/dates.csv \
        --track-outpath $this_test_out/hurricane-track.dat \
        --swath-outpath $this_test_out/windswath \
        --station-data-outpath $this_test_out/stations.nc \
        --station-location-outpath $this_test_out/stations.csv \
        --past-forecast \
        --hours-before-landfall "$hr_prelandfall" \
        --lead-times "$L_LEADTIMES_DATASET" \
        --preprocessed-tracks-dir "$L_TRACK_DIR" \
        $storm $year
    
    this_test_out=$run_dir/info_w_leadjson_$hr_prelandfall
    mkdir $this_test_out
    python $SINGULARITY_ROOT/info/files/hurricane_data.py \
        --date-range-outpath $this_test_out/dates.csv \
        --track-outpath $this_test_out/hurricane-track.dat \
        --swath-outpath $this_test_out/windswath \
        --station-data-outpath $this_test_out/stations.nc \
        --station-location-outpath $this_test_out/stations.csv \
        --past-forecast \
        --hours-before-landfall "$hr_prelandfall" \
        --lead-times "$L_LEADTIMES_DATASET" \
        $storm $year

    this_test_out=$run_dir/info_w_leadjson_24
    mkdir $this_test_out
    python $SINGULARITY_ROOT/info/files/hurricane_data.py \
        --date-range-outpath $this_test_out/dates.csv \
        --track-outpath $this_test_out/hurricane-track.dat \
        --swath-outpath $this_test_out/windswath \
        --station-data-outpath $this_test_out/stations.nc \
        --station-location-outpath $this_test_out/stations.csv \
        --past-forecast \
        --hours-before-landfall 24 \
        --lead-times "$L_LEADTIMES_DATASET" \
        $storm $year

    this_test_out=$run_dir/info_w_preptrack_$hr_prelandfall
    mkdir $this_test_out
    python $SINGULARITY_ROOT/info/files/hurricane_data.py \
        --date-range-outpath $this_test_out/dates.csv \
        --track-outpath $this_test_out/hurricane-track.dat \
        --swath-outpath $this_test_out/windswath \
        --station-data-outpath $this_test_out/stations.nc \
        --station-location-outpath $this_test_out/stations.csv \
        --past-forecast \
        --hours-before-landfall "$hr_prelandfall" \
        --preprocessed-tracks-dir "$L_TRACK_DIR" \
        $storm $year

    this_test_out=$run_dir/info_w_leadjson_besttrack_$hr_prelandfall
    mkdir $this_test_out
    python $SINGULARITY_ROOT/info/files/hurricane_data.py \
        --date-range-outpath $this_test_out/dates.csv \
        --track-outpath $this_test_out/hurricane-track.dat \
        --swath-outpath $this_test_out/windswath \
        --station-data-outpath $this_test_out/stations.nc \
        --station-location-outpath $this_test_out/stations.csv \
        --hours-before-landfall "$hr_prelandfall" \
        --lead-times "$L_LEADTIMES_DATASET" \
        $storm $year

    this_test_out=$run_dir/info_w_besttrack_$hr_prelandfall
    mkdir $this_test_out
    python $SINGULARITY_ROOT/info/files/hurricane_data.py \
        --date-range-outpath $this_test_out/dates.csv \
        --track-outpath $this_test_out/hurricane-track.dat \
        --swath-outpath $this_test_out/windswath \
        --station-data-outpath $this_test_out/stations.nc \
        --station-location-outpath $this_test_out/stations.csv \
        --hours-before-landfall "$hr_prelandfall" \
        $storm $year
}

$1
