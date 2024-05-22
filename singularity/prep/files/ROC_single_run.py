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
    storm_name = args.storm_name.capitalize()
    storm_year = args.storm_year
    leadtime = args.leadtime
    prob_nc_path = Path(args.prob_nc_path)
    obs_df_path = Path(args.obs_df_path)
    save_dir = args.save_dir

    # *.nc file coordinates
    thresholds_ft = [3, 6, 9]  # in ft
    thresholds_m = [round(i * 0.3048, 4) for i in thresholds_ft]  # convert to meter
    sources = ['model', 'surrogate']
    probabilities = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # attributes of input files
    prediction_variable = 'probabilities'
    obs_attribute = 'Elev_m_xGEOID20b'

    # search criteria
    max_distance = 1000  # [in meters] to set distance_upper_bound
    max_neighbors = 10  # to set k

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
    Event_name = f'{storm_name}_{storm_year}'
    df_obs_storm = df_obs[df_obs.Event == Event_name]
    obs_coordinates = stack_station_coordinates(
        df_obs_storm.Longitude.values, df_obs_storm.Latitude.values
    )

    # Load probabilities.nc file
    ds_prob = xr.open_dataset(prob_nc_path)

    gdf_countries = gpd.GeoSeries(
        NaturalEarthFeature(category='physical', scale='10m', name='land',).geometries(),
        crs=4326,
    )

    # Loop through thresholds and sources and find corresponding values from probabilities.nc
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
            plot_probabilities(
                df_obs_storm,
                f'{source}_prob',
                gdf_countries,
                f'Probability of {source} exceeding {thresholds_ft[threshold_count]} ft \n {storm_name}, {storm_year}, {leadtime}-hr leadtime',
                os.path.join(
                    save_dir,
                    f'prob_{source}_above_{thresholds_ft[threshold_count]}ft_{storm_name}_{storm_year}_{leadtime}-hr.png',
                ),
            )

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
            storm=[storm_name],
            leadtime=[leadtime],
            source=sources,
            prob=probabilities,
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
        ),
    )
    ds_ROC.to_netcdf(
        os.path.join(save_dir, f'{storm_name}_{storm_year}_{leadtime}hr_leadtime_POD_FAR.nc')
    )

    # plot ROC curves
    marker_list = ['s', 'x']
    linestyle_list = ['dashed', 'dotted']
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
            plt.plot(
                FAR_arr[threshold_count, 0, 0, source_count, :],
                POD_arr[threshold_count, 0, 0, source_count, :],
                label=f'{source}',
                marker=marker_list[source_count],
                linestyle=linestyle_list[source_count],
                markersize=5,
            )
        plt.legend()
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Probability of Detection')

        plt.title(
            f'{storm_name}_{storm_year}, {leadtime}-hr leadtime, {threshold} ft threshold'
        )
        plt.savefig(
            os.path.join(
                save_dir, f'ROC_{storm_name}_{leadtime}hr_leadtime_{threshold}_ft.png'
            )
        )
        plt.close()


def entry():
    parser = argparse.ArgumentParser()

    parser.add_argument('--storm_name', help='name of the storm', type=str)

    parser.add_argument('--storm_year', help='year of the storm', type=int)

    parser.add_argument('--leadtime', help='OFCL track leadtime hr', type=int)

    parser.add_argument('--prob_nc_path', help='path to probabilities.nc', type=str)

    parser.add_argument('--obs_df_path', help='Path to observations dataframe', type=str)

    # optional
    parser.add_argument(
        '--save_dir', help='directory for saving analysis', default=os.getcwd(), type=str
    )

    main(parser.parse_args())


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    entry()
