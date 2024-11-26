import argparse
import logging
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
from cartopy.feature import NaturalEarthFeature
from geodatasets import get_path
from sklearn.calibration import calibration_curve

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

pd.options.mode.copy_on_write = True


def stack_station_coordinates(x, y):
    """
    Create numpy.column_stack based on
    coordinates of observation points
    """
    coord_combined = np.column_stack([x, y])
    return coord_combined


def create_search_tree(longitude, latitude):
    """
    Create scipy.spatial.CKDTree based on Lat. and Long.
    """
    long_lat = np.column_stack((longitude.T.ravel(), latitude.T.ravel()))
    tree = sp.spatial.cKDTree(long_lat)
    return tree


def find_nearby_prediction(ds, variable, indices):
    """
    Reads netcdf file, target variable, and indices
    Returns max value among corresponding indices for each point 
    """
    obs_count = indices.shape[0]  # total number of search/observation points
    max_prediction_index = len(ds.node.values)  # total number of nodes

    prediction_prob = np.zeros(obs_count)  # assuming all are dry (probability of zero)

    for obs_point in range(obs_count):
        idx_arr = np.delete(
            indices[obs_point], np.where(indices[obs_point] == max_prediction_index)[0]
        )  # len is length of surrogate model array
        val_arr = ds[variable].values[idx_arr]
        val_arr = np.nan_to_num(val_arr)  # replace nan with zero (dry node)

        # # Pick the nearest non-zero probability (option #1)
        # for val in val_arr:
        #     if val > 0.0:
        #         prediction_prob[obs_point] = round(val,4) #round to 0.1 mm
        #         break

        # pick the largest value (option #2)
        if val_arr.size > 0:
            prediction_prob[obs_point] = val_arr.max()
    return prediction_prob


def plot_probabilities(df, prob_column, gdf_countries, title, save_name):
    """
    plot probabilities of exceeding given threshold at obs. points
    """
    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(10, 10 / 1.6)

    plt.scatter(x=df.Longitude, y=df.Latitude, vmin=0, vmax=1.0, c=df[prob_column])
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()

    gdf_countries.plot(color='lightgrey', ax=axis, zorder=-5)

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    plt.colorbar(shrink=0.75)
    plt.title(title)
    plt.savefig(save_name)
    plt.close()


def calculate_hit_miss(df, obs_column, prob_column, threshold, probability):
    """
    Reads dataframe with two columns for obs_elev, and probabilities
    returns hit/miss/... based on user-defined threshold & probability
    """
    hit = len(df[(df[obs_column] >= threshold) & (df[prob_column] >= probability)])
    miss = len(df[(df[obs_column] >= threshold) & (df[prob_column] < probability)])
    false_alarm = len(df[(df[obs_column] < threshold) & (df[prob_column] >= probability)])
    correct_neg = len(df[(df[obs_column] < threshold) & (df[prob_column] < probability)])

    return hit, miss, false_alarm, correct_neg


def calculate_POD_FAR(hit, miss, false_alarm, correct_neg):
    """
    Reads hit, miss, false_alarm, and correct_neg
    returns POD and FAR
    default POD and FAR are np.nan
    """
    POD = np.nan
    FAR = np.nan
    try:
        POD = round(hit / (hit + miss), 4)  # Probability of Detection
    except ZeroDivisionError:
        pass
    try:
        FAR = round(false_alarm / (false_alarm + correct_neg), 4)  # False Alarm Rate
    except ZeroDivisionError:
        pass
    return POD, FAR


def main(args):
    storm = args.storm.capitalize()
    year = args.year
    leadtime = args.leadtime
    obs_df_path = Path(args.obs_df_path)
    ensemble_dir = Path(args.ensemble_dir)
    output_directory = args.output_dir
    suffix = args.output_suffix
    plot_prob_map = args.plot_prob_map
    psurge_far_path = args.psurge_far_path
    psurge_pod_path = args.psurge_pod_path

    input_directory = ensemble_dir / 'analyze/linear_k1_p1_n0.025'
    prob_nc_path = input_directory / 'probabilities.nc'
    if output_directory is None:
        output_directory = input_directory

    if leadtime == -1:
        leadtime = 48

    # *.nc file coordinates
    thresholds_ft = [3, 4, 5, 6, 9]  # in ft
    thresholds_m = [round(i * 0.3048, 4) for i in thresholds_ft]  # convert to meter
    sources = ['model', 'surrogate']
    probabilities = list(np.linspace(0, 1, 101))

    # attributes of input files
    prediction_variable = 'probabilities'
    obs_attribute = 'Elev_m_xGEOID20b'

    # search criteria
    max_distance = 1000  # [in meters] to set distance_upper_bound
    max_neighbors = 10  # to set k

    # creating arrays for ROC and TS plots
    blank_arr = np.empty((len(thresholds_ft), 1, 1, len(sources), len(probabilities)))
    blank_arr[:] = np.nan
    hit_arr = blank_arr.copy()
    miss_arr = blank_arr.copy()
    false_alarm_arr = blank_arr.copy()
    correct_neg_arr = blank_arr.copy()
    POD_arr = blank_arr.copy()
    FAR_arr = blank_arr.copy()

    # Load obs file, extract storm obs points and coordinates
    df_obs = pd.read_csv(obs_df_path)
    # Get first matching name (first landfall)
    CSV_Event_name = df_obs[df_obs.Event.str.fullmatch(f'{storm}\w*_{year}')].Event.iloc[0]
    NC_Event_name = CSV_Event_name[:CSV_Event_name.find('_')]
    df_obs_storm = df_obs[df_obs.Event == CSV_Event_name]
    obs_coordinates = stack_station_coordinates(
        df_obs_storm.Longitude.values, df_obs_storm.Latitude.values
    )

    # creating arrays for reliability diagram
    blank_arr = np.empty((len(thresholds_ft), 1, 1, len(sources), len(df_obs_storm)))
    blank_arr[:] = np.nan
    obs_true_arr = blank_arr.copy()
    pred_prob_arr = blank_arr.copy()

    # Load Psurge results
    ds_psurge = None
    if not (psurge_far_path is None or psurge_pod_path is None):
        ds_psurge = xr.open_mfdataset(
            [psurge_far_path, psurge_pod_path],
            combine='nested',
        )

    # Load probabilities.nc file
    ds_prob = xr.open_dataset(prob_nc_path)

    gdf_countries = gpd.read_file(get_path('naturalearth land'))

    threshold_count = -1
    for threshold in thresholds_m:
        threshold_count += 1
        source_count = -1
        for source in sources:
            source_count += 1
            ds_temp = ds_prob.sel(level=threshold, source=source)
            tree = create_search_tree(ds_temp.x.values, ds_temp.y.values)
            dist, indices = tree.query(
                obs_coordinates, k=max_neighbors, distance_upper_bound=max_distance * 1e-5
            )  # 0.01 is equivalent to 1000 m
            prediction_prob = find_nearby_prediction(
                ds=ds_temp, variable=prediction_variable, indices=indices
            )
            df_obs_storm[f'{source}_prob'] = prediction_prob

            # Plot probabilities at obs. points
            if plot_prob_map:
                plot_probabilities(
                    df_obs_storm,
                    f'{source}_prob',
                    gdf_countries,
                    f'Probability of {source} exceeding {thresholds_ft[threshold_count]} ft \n {storm}, {year}, {leadtime}-hr leadtime',
                    os.path.join(
                        output_directory,
                        f'prob_{source}_above_{thresholds_ft[threshold_count]}ft_{storm}_{year}_{leadtime}-hr_{suffix}.png',
                    ),
                )

            # Enter observed above threshold and prediction probability into array
            obs_true_arr[threshold_count, 0, 0, source_count, :] = (
                df_obs_storm[obs_attribute] > threshold
            )
            pred_prob_arr[threshold_count, 0, 0, source_count, :] = prediction_prob

            # Loop through probabilities: calculate hit/miss/... & POD/FAR
            prob_count = -1
            for prob in probabilities:
                prob_count += 1
                hit, miss, false_alarm, correct_neg = calculate_hit_miss(
                    df_obs_storm, obs_attribute, f'{source}_prob', threshold, prob
                )
                hit_arr[threshold_count, 0, 0, source_count, prob_count] = hit
                miss_arr[threshold_count, 0, 0, source_count, prob_count] = miss
                false_alarm_arr[threshold_count, 0, 0, source_count, prob_count] = false_alarm
                correct_neg_arr[threshold_count, 0, 0, source_count, prob_count] = correct_neg

                pod, far = calculate_POD_FAR(hit, miss, false_alarm, correct_neg)
                POD_arr[threshold_count, 0, 0, source_count, prob_count] = pod
                FAR_arr[threshold_count, 0, 0, source_count, prob_count] = far

    ds_ROC = xr.Dataset(
        coords=dict(
            threshold=thresholds_ft,
            storm=[storm],
            leadtime=[leadtime],
            source=sources,
            prob=probabilities,
            points=range(len(df_obs_storm)),
        ),
        data_vars=dict(
            hit=(['threshold', 'storm', 'leadtime', 'source', 'prob'], hit_arr),
            miss=(['threshold', 'storm', 'leadtime', 'source', 'prob'], miss_arr),
            false_alarm=(
                ['threshold', 'storm', 'leadtime', 'source', 'prob'],
                false_alarm_arr,
            ),
            correct_neg=(
                ['threshold', 'storm', 'leadtime', 'source', 'prob'],
                correct_neg_arr,
            ),
            POD=(['threshold', 'storm', 'leadtime', 'source', 'prob'], POD_arr),
            FAR=(['threshold', 'storm', 'leadtime', 'source', 'prob'], FAR_arr),
            obs_true=(['threshold', 'storm', 'leadtime', 'source', 'points'], obs_true_arr),
            pred_prob=(['threshold', 'storm', 'leadtime', 'source', 'points'], pred_prob_arr),
        ),
    )
    ds_ROC.to_netcdf(
        os.path.join(
            output_directory, f'{storm}_{year}_{leadtime}hr_probabilistic_evaluation_stats_{suffix}.nc'
        )
    )

    # plot ROC curves
    colormarker_list = ['bd', 'kx']
    linestyle_list = ['--', '-']
    threshold_count = -1
    for threshold in thresholds_ft:
        threshold_count += 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axline(
            (0.0, 0.0), (1.0, 1.0), linestyle='--', color='grey', label='random prediction'
        )
        source_count = -1
        for source in sources:
            source_count += 1
            AUC = abs(
                np.trapz(
                    POD_arr[threshold_count, 0, 0, source_count, :],
                    x=FAR_arr[threshold_count, 0, 0, source_count, :],
                )
            )
            label = f'{source}, AUC={AUC:.2f}'
            colormarker = colormarker_list[source_count]
            if source == 'surrogate':
                best_res = (
                    POD_arr[threshold_count, 0, 0, source_count, :]
                    - FAR_arr[threshold_count, 0, 0, source_count, :]
                ).argmax()
                label = f'{label}, x @ p({probabilities[best_res]:.2f})'
                plt.plot(
                    FAR_arr[threshold_count, 0, 0, source_count, best_res],
                    POD_arr[threshold_count, 0, 0, source_count, best_res],
                    'kX',
                )
            plt.plot(
                FAR_arr[threshold_count, 0, 0, source_count, :],
                POD_arr[threshold_count, 0, 0, source_count, :],
                colormarker,
                label=label,
                linestyle=linestyle_list[source_count],
                markersize=5,
            )
        if ds_psurge is not None and threshold in ds_psurge.threshold:
            psurge_far = ds_psurge.sel(
                    version='v3pt0Apr062023_kdtree',
                    threshold=threshold,
                    storm=NC_Event_name,
                    leadtime=leadtime).FARate.values
            psurge_pod = ds_psurge.sel(
                    version='v3pt0Apr062023_kdtree',
                    threshold=threshold,
                    storm=NC_Event_name,
                    leadtime=leadtime).POD.values
            AUC = abs(np.trapz(psurge_pod, x=psurge_far))
            label = f'psurge, AUC={AUC:.2f}'
            plt.plot(
                psurge_far,
                psurge_pod,
                'ro',
                label=label,
                linestyle=':',
                markersize=4,
            )
        plt.legend(loc='lower right')
        npos = int(hit_arr[threshold_count, 0, 0, 0, 0])
        nneg = int(false_alarm_arr[threshold_count, 0, 0, 0, 0])
        plt.xlabel(f'False Alarm Rate, N={nneg}')
        plt.ylabel(f'Probability of Detection, N={npos}')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.grid(True)

        plt.title(
            f'{storm}_{year}, {leadtime}-hr leadtime, {threshold} ft threshold: N={len(df_obs_storm)}'
        )
        plt.savefig(
            os.path.join(
                output_directory,
                f'ROC_{storm}_{year}_{leadtime}hr_leadtime_{threshold}_ft_{suffix}.png',
            )
        )
        plt.close()

    # plot TS curves
    threshold_count = -1
    for threshold in thresholds_ft:
        threshold_count += 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        source_count = -1
        for source in sources:
            source_count += 1
            # TS = hits / (hits + misses + false alarms)
            TS = hit_arr[threshold_count, 0, 0, source_count, :] / (
                hit_arr[threshold_count, 0, 0, source_count, :]
                + miss_arr[threshold_count, 0, 0, source_count, :]
                + false_alarm_arr[threshold_count, 0, 0, source_count, :]
            )

            plt.plot(
                probabilities,
                TS,
                colormarker_list[source_count][0],
                label=f'{source}',
                linestyle=linestyle_list[source_count],
            )
        plt.legend(loc='upper right')
        plt.ylabel(f'Threat Score')
        plt.xlabel(f'probability of exceedance')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.grid(True)

        plt.title(
            f'{storm}_{year}, {leadtime}-hr leadtime, {threshold} ft threshold: N={len(df_obs_storm)}'
        )
        plt.savefig(
            os.path.join(
                output_directory,
                f'TS_{storm}_{year}_{leadtime}hr_leadtime_{threshold}_ft_{suffix}.png',
            )
        )
        plt.close()

    # plot reliability curves
    threshold_count = -1
    for threshold in thresholds_ft:
        threshold_count += 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axline(
            (0.0, 0.0), (1.0, 1.0), linestyle='--', color='grey', label='perfect reliability'
        )
        source_count = -1
        for source in sources:
            source_count += 1
            true_prob, pred_prob = calibration_curve(
                obs_true_arr[threshold_count, 0, 0, source_count, :],
                pred_prob_arr[threshold_count, 0, 0, source_count, :],
                n_bins=5,
                strategy='uniform',
            )
            plt.plot(
                pred_prob,
                true_prob,
                colormarker_list[source_count],
                label=f'{source}',
                linestyle=linestyle_list[source_count],
                markersize=5,
            )

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.legend(loc='lower right')
        plt.xlabel(f'Predicted probability of exceedance')
        plt.ylabel(f'Observed fraction of exceedances')
        plt.grid(True)

        plt.title(
            f'{storm}_{year}, {leadtime}-hr leadtime, {threshold} ft threshold: N={len(df_obs_storm)}'
        )
        plt.savefig(
            os.path.join(
                output_directory,
                f'REL_{storm}_{year}_{leadtime}hr_leadtime_{threshold}_ft_{suffix}.png',
            )
        )
        plt.close()


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--storm', help='name of the storm', type=str)
    parser.add_argument('--year', help='year of the storm', type=int)
    parser.add_argument('--leadtime', help='OFCL track leadtime hr', type=int)
    parser.add_argument('--obs_df_path', help='path to NHC obs data', type=str)
    parser.add_argument('--ensemble-dir', help='path to ensemble.dir', type=str)

    # optional
    parser.add_argument(
        '--output-dir',
        help='directory to save the outputs of this function',
        default=None,
        type=str,
    )

    parser.add_argument(
        '--plot_prob_map',
        help='plot the prediction probability at observations maps',
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        '--psurge-far-path',
        help='path to NHC PSurge FAR results',
        type=Path
    )
    parser.add_argument(
        '--psurge-pod-path',
        help='path to NHC PSurge POD results',
        type=Path
    )
    parser.add_argument(
        '--output-suffix',
        help='prefix to be used for path of output files',
        type=str
    )

    main(parser.parse_args())


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    cli()
