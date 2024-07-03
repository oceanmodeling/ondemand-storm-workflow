import argparse
import logging
import os
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

from cartopy.feature import NaturalEarthFeature
from datetime import datetime,timedelta
from matplotlib.dates import DateFormatter
from pathlib import Path 
from pyschism.mesh import Hgrid
from stormevents import StormEvent
from searvey.coops import coops_product_within_region, coops_stations_within_region
from shapely.geometry import MultiPolygon
from shapely.ops import polygonize

_logger = logging.getLogger()
warnings.filterwarnings("ignore", category=DeprecationWarning)

############## functions ##############
def format_datetime_for_coops(storm_datetime):
    return datetime(storm_datetime.year, storm_datetime.month, storm_datetime.day, storm_datetime.hour, storm_datetime.minute)


def get_time_period_of_interest(mid_day, no_of_days):
    start_date = mid_day + timedelta(days=-no_of_days, hours=12)
    end_date = mid_day + timedelta(days=no_of_days, hours=12)
    return  start_date, end_date
    

def get_corr(obs,sim):
    return obs.corr(sim)


def get_mae(obs,sim):
    mae = ((np.abs(obs-sim)).sum()) / len(obs)
    # mae = (np.abs(obs-sim)).mean()
    return mae


def get_r2(obs,sim):
    nom = ((obs-obs.mean()) * (sim-sim.mean())).sum()
    den_1 = np.sqrt(((obs-obs.mean())**2).sum())
    den_2 = np.sqrt(((sim-sim.mean())**2).sum())
    return (nom/(den_1*den_2))**2


def get_rmse(obs,sim):
    # rmse = np.sqrt(((obs-sim)**2).mean())
    dif = (obs-sim)**2
    rmse = np.sqrt(dif.sum()/len(obs))
    return rmse


def get_bias(obs,sim):
    return (sim-obs).mean()


def get_rel_bias(obs,sim):
    rel_bias = get_bias(obs,sim) / obs.mean()
    return rel_bias


def calc_stats(obs,sim):
    df_stats = pd.DataFrame(columns=['NOS_ID', 'STD_REF', 'STD_MOD', 'CORR', 'MAE', 'R2', 'RMSE', 'BIAS', 'R_BIAS'])
    for idx in obs.columns:
        ref_std = '%.2f' % round(obs[idx].std(ddof=1),2)
        mod_std = '%.2f' % round(sim[idx].std(ddof=1),2)
        temp_corr = '%.2f' % round(get_corr(obs[idx],sim[idx]),2)
        temp_mae = '%.2f' % round(get_mae(obs[idx],sim[idx]),2)
        temp_r2 = '%.2f' % round(get_r2(obs[idx],sim[idx]),2)
        temp_rmse = '%.2f' % round(get_rmse(obs[idx],sim[idx]),2)
        temp_bias = '%.2f' % round(get_bias(obs[idx],sim[idx]),2)
        temp_rel_bias = '%.2f' % round(get_rel_bias(obs[idx],sim[idx]),2)
        
        df_temp = pd.DataFrame([[idx, ref_std, mod_std, temp_corr, temp_mae,  temp_r2, temp_rmse, temp_bias, temp_rel_bias]],
                               columns=['NOS_ID', 'STD_REF', 'STD_MOD', 'CORR', 'MAE', 'R2', 'RMSE', 'BIAS', 'R_BIAS'])
        df_stats = pd.concat([df_stats,df_temp], axis=0, ignore_index=True)
    return df_stats


def plot_isobar_domains(storm_track, plot_date, save_dir, countries_map):
    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(12, 12 / 1.618)

    axis.plot(*storm_track.wind_swaths(wind_speed=34)['BEST'][plot_date].exterior.xy, 
              c='limegreen', label='34 kt')
    axis.plot(*storm_track.wind_swaths(wind_speed=50)['BEST'][plot_date].exterior.xy, 
              c='blue', label='50 kt')
    axis.plot(*storm_track.wind_swaths(wind_speed=64)['BEST'][plot_date].exterior.xy, 
              c='red', label='64 kt')
    axis.plot(*storm_track.linestrings['BEST'][plot_date].xy, 
              c='black',label='BEST track')
    
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    countries_map.plot(color='lightgrey', ax=axis, zorder=-1)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.legend()
    axis.set_title(f'{storm_track.name}_{storm_track.year} BEST track windswatch')
    
    figure.savefig(os.path.join(save_dir, f'{storm_track.name}_{storm_track.year}_BEST_track_windswatch.png'))

    
def plot_coops_stations(storm_name, storm_year, coops_tidal_pred, save_dir, countries_map):
    x_vals = coops_tidal_pred['x'].values
    y_vals = coops_tidal_pred['y'].values
    nos_ids = coops_tidal_pred['nos_id'].values
    
    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(12, 12 / 1.618)

    for idx in range(len(nos_ids)):
        axis.scatter(x_vals[idx],y_vals[idx], label=nos_ids[idx])
    
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    countries_map.plot(color='lightgrey', ax=axis, zorder=-1)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.legend()
    axis.set_title(f'COOPS tidal stations within 50kt isotach of {storm_name}_{storm_year}')
    
    figure.savefig(os.path.join(save_dir, f'{storm_name}_{storm_year}_COOPS_tidal_stations.png'))


def coops_dataset_to_dataframe(coops_dataset): #, stations=None):
    df_coops, ij = [], 0
    for idx in coops_dataset.coords['nos_id'].values:
        df_temp = pd.DataFrame({'date': coops_dataset.coords['t'].values,
                                idx: coops_dataset.v.values[ij,:],
                               }).set_index('date')
        df_coops.append(df_temp)
        ij = ij+1
    df_coops = pd.concat(df_coops, axis=1)       
    return df_coops


def adjust_coops_water_level(df_coops):
    df_coops_adjusted = pd.DataFrame()
    for coops_idx in df_coops.columns:
        df_coops_adjusted[coops_idx] = df_coops[coops_idx] - df_coops[coops_idx].values.mean()
    return df_coops_adjusted


def get_stations_coordinates(coops_dataset):
    coord_x = coops_dataset.x.values
    coord_y = coops_dataset.y.values
    coord_combined = np.column_stack([coord_x, coord_y])
    return coord_combined


def find_stations_indices(coordinates, hgridfile):
    longitude = hgridfile.x.T
    latitude = hgridfile.y.T
    long_lat = np.column_stack((longitude.ravel(),latitude.ravel()))
    tree = sp.spatial.cKDTree(long_lat)
    dist,idx = tree.query(coordinates,k=1)
    ind = np.column_stack(np.unravel_index(idx,longitude.shape))
    return [i for i in ind]
    
    
def get_schism_elevation_df(schism_output, station_indices_array, nos_ids):
    time_array = schism_output.time.values
    var_name = 'elevation'
    
    df_sim, ij = [], 0
    for idx in nos_ids:
        df_temp = pd.DataFrame({'date': time_array,
                                idx: schism_output[var_name][:,int(station_indices_array[ij])].values,
                               }).set_index('date')
        df_sim.append(df_temp)
        ij = ij+1 
    df_sim = pd.concat(df_sim, axis=1)
    return df_sim


def plot_timeseries(df_obs, df_sim, station_lookup, storm_name, storm_year, save_dir):
    n_plots = len(df_obs.columns)

    fig_wspace, fig_hspace = 0.2, 0.2
    
    if n_plots <= 6:
        row, col = n_plots, 1
    else:
        row, col = int(np.ceil(n_plots/2)), 2

    fig_width = 8*(col) + fig_wspace *(col-1)
    fig_height = 2*(row) + fig_hspace *(row-1)
    
    fig, axis = plt.subplots(row , col, figsize=(fig_width, fig_height), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = fig_hspace, wspace=fig_wspace)
    axis = axis.ravel()
    idx_plt = 0
    
    for idx in df_obs.columns: 
        axis[idx_plt].plot(df_obs[idx], linewidth=0.7, label='COOPS tidal')
        axis[idx_plt].plot(df_sim[idx], linewidth=0.7, label='SCHISM tidal')            

        axis[idx_plt].grid(axis = 'both', color = 'gray', linestyle = '-', linewidth = 0.75, alpha=0.15)
        axis[idx_plt].tick_params(axis="both",direction="in")  #, pad=0
        plt.setp(axis[idx_plt].xaxis.get_majorticklabels(), rotation=90)
        
        # format x-labels
        plt.gcf().autofmt_xdate()
        date_form = DateFormatter("%b-%d, %H:%M")
        axis[idx_plt].xaxis.set_major_formatter(date_form)        
        axis[idx_plt].set_ylim([-1.5,1.5])
    
        station_label = f'{idx} ({station_lookup[station_lookup.nos_id==int(idx)].name.values[0]})'
        axis[idx_plt].text(0.5, 0.975, station_label,
                           horizontalalignment='center', verticalalignment='top',
                           transform=axis[idx_plt].transAxes, size=9,weight='bold')        
        idx_plt +=1

    # add legend:
    axis[-1].legend(loc="lower right", ncol=2)
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel('WSE [m]', size=11,weight='bold')
    plt.title(f'{storm_name}_{storm_year} Tidal Validation \n From {df_obs.index[0]} To {df_obs.index[0]}', size=15)
    
    plt.savefig(os.path.join(save_dir, f'{storm_name}_{storm_year}_timeseries_tidal.png'))#, dpi=250)

    
def plot_scatter(df_obs, df_sim, df_stats, station_lookup, storm_name, storm_year, save_dir):
    
    n_plots = len(df_obs.columns)
    row, col = int(np.ceil(n_plots/3)), 3
    fig_wspace, fig_hspace = 0.2, 0.2
    fig_width = 5*(col) + fig_wspace *(col-1)
    fig_height = 5*(row) + fig_hspace *(row-1)
    
    fig, axis = plt.subplots(row , col, figsize=(fig_width, fig_height), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = fig_hspace, wspace=fig_wspace)
    
    axis = axis.ravel()
    idx_plt = 0
    
    for idx in df_obs.columns: 
        axis[idx_plt].scatter(df_obs[idx], df_sim[idx], s=4, label=idx)
        axis[idx_plt].axline((-1.5,-1.5), (1.5,1.5), linestyle='--', color='grey')       
        # axis[idx_plt].legend(loc="upper left")
        axis[idx_plt].tick_params(axis="both",direction="in")  #, pad=0
        plt.setp(axis[idx_plt].xaxis.get_majorticklabels(), rotation=0)
        
        station_label = f'{idx} ({station_lookup[station_lookup.nos_id==int(idx)].name.values[0]})'
        axis[idx_plt].text(0.5, 0.975, station_label,
                   horizontalalignment='center', verticalalignment='top',
                   transform=axis[idx_plt].transAxes, size=9,weight='bold')
        
        axis[idx_plt].text(0.05, 0.85, f"Cor={df_stats[df_stats.NOS_ID==str(idx)]['CORR'].values[0]}",
                           horizontalalignment='left', verticalalignment='center',
                           transform=axis[idx_plt].transAxes, size=10)
        
        axis[idx_plt].text(0.05, 0.79, f"R2={df_stats[df_stats.NOS_ID==str(idx)]['R2'].values[0]}",
                           horizontalalignment='left', verticalalignment='center',
                           transform=axis[idx_plt].transAxes, size=10)
                           
        axis[idx_plt].text(0.05, 0.73, f"MAE={df_stats[df_stats.NOS_ID==str(idx)]['MAE'].values[0]}",
                           horizontalalignment='left', verticalalignment='center',
                           transform=axis[idx_plt].transAxes, size=10)

        axis[idx_plt].text(0.05, 0.67, f"RMSE={df_stats[df_stats.NOS_ID==str(idx)]['RMSE'].values[0]}",
                           horizontalalignment='left', verticalalignment='center',
                           transform=axis[idx_plt].transAxes, size=10)
                           
        axis[idx_plt].text(0.05, 0.61, f"BIAS={df_stats[df_stats.NOS_ID==str(idx)]['BIAS'].values[0]}",
                           horizontalalignment='left', verticalalignment='center',
                           transform=axis[idx_plt].transAxes, size=10)

        axis[idx_plt].text(0.05, 0.55, f"R_BIAS={df_stats[df_stats.NOS_ID==str(idx)]['R_BIAS'].values[0]}",
                           horizontalalignment='left', verticalalignment='center',
                           transform=axis[idx_plt].transAxes, size=10)
        if idx_plt%3==0:
            axis[idx_plt].set_ylabel('SCHISM tidal simulation [m]')
            
        idx_plt +=1
    
    axis[-1].set_xlabel('COOPS tidal prediction [m]')
    axis[-2].set_xlabel('COOPS tidal prediction [m]')
    axis[-3].set_xlabel('COOPS tidal prediction [m]')
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.title(f'{storm_name}_{storm_year} Tidal Validation \n From {df_obs.index[0]} To {df_obs.index[0]}', size=15)
    plt.savefig(os.path.join(save_dir, f'{storm_name}_{storm_year}_scatters_with_stats.png'))#, dpi=250)

    
def plot_taylor_diagram(df_stats, storm_name, storm_year, save_dir):
    
    fig = plt.figure(1,figsize=(6,6))
    fig.clf()
    fig.suptitle(f'{storm_name}_{storm_year} \n ' u"\u2598" ': SCHISM sim. ; *: COOPS obs.')
    colors = plt.matplotlib.cm.jet(np.linspace(0,1,len(df_stats.index)))
    dia = TaylorDiagram(df_stats['STD_REF'].values, fig=fig, rect=111)
    
    # Add samples to Taylor diagram
    for index,row in df_stats.iterrows():
        dia.add_sample(float(row['STD_MOD']), float(row['CORR']), 
                       marker='s', ls='', c=colors[index],
                       label=row['NOS_ID'])
        dia.add_sample(float(row['STD_REF']), 1.0 ,
                       marker='*', ls='', c=colors[index])
    # Add a figure legend    
    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='upper right')
    plt.savefig(os.path.join(save_dir, f'{storm_name}_{storm_year}_TaylorDiagram.png'))
    

########################################
class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, refstd, fig=None, rect=111, label='_'):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes. refstd is
        the reference standard deviation to be compared to.
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.concatenate((np.arange(10)/10.,[0.95,0.99]))
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str,rlocs))))

        # Standard deviation axis extent
        self.smin = 0
        self.smax = 1.5*float(max(refstd))     #1.5*self.refstd (updated by Fariborz)

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=(0,np.pi/2, # 1st quadrant
                                                     self.smin,self.smax),
                                           grid_locator1=gl1,
                                           tick_formatter1=tf1,
                                           )
        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom") # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")   # "Y axis"
        ax.axis["right"].label.set_text("Standard deviation")

        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)         # Useless
        
        # Contours along standard deviations
        ax.grid(False)

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates
       
        for ip in range (len(rlocs)):
            x=[self.smin,self.smax]
            y=[np.arccos(rlocs[ip]),np.arccos(rlocs[ip])]
            self.ax.plot(y,x, 'grey',linewidth=0.25)
        
        # Collect sample points for latter use (e.g. legend)
        #self.samplePoints = [l]                       rem by saeed
        self.samplePoints = []                        # add by saeed
    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample (stddev,corrcoeff) to the Taylor diagram. args
        and kwargs are directly propagated to the Figure.plot
        command."""

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs) # (theta,radius)
        self.samplePoints.append(l)
        
        # if ref:
        #     t = np.linspace(0, np.pi/2) # add by saeed
        #     r = np.zeros_like(t) + stddev # add by saeed
        #     self.ax.plot(t,r, 'grey' ,linewidth=0.25,label='_') # add by saeed

        return l

#     def add_contours(self,levels,data_std, **kwargs):
#         """Add constant centered RMS difference contours."""

#         rs,ts = np.meshgrid(np.linspace(self.smin,self.smax),
#                             np.linspace(0,np.pi/2))
#         # Compute centered RMS difference
#         rms = np.sqrt(data_std**2 + rs**2 - 2*data_std*rs*np.cos(ts))
        
#         contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

#         return contours   
    
    
###############################
def main(args):

    name_of_storm = args.storm_name.upper()
    year_of_storm = args.storm_year
    target_date = datetime.strptime(args.mid_date, '%Y-%m-%d')
    hgrid_directory = Path(args.grid_dir)
    output_directory = Path(args.output_dir)
    save_directory = args.save_dir
       
    hgrid_file_path = hgrid_directory / 'hgrid.gr3'
    if not hgrid_file_path.exists():
        raise FileNotFoundError(f'{hgrid_file_path} was not found!')
    
    output_file_path = output_directory / 'out2d_1.nc'
    if not output_file_path.exists():
        raise FileNotFoundError(f"{output_file_path} was not found!")

    print(f"{name_of_storm}_{year_of_storm}")
    # _logger.info(f'Retrieving {name_of_storm}_{year_of_storm} from stormevents ...')
    
    gdf_countries = gpd.GeoSeries(NaturalEarthFeature(category='physical', scale='10m', name='land').geometries(), crs=4326)

    # Load storm and obtain isobars from stormevent
    storm = StormEvent(name_of_storm, year_of_storm)
    storm_best_track = storm.track()
    
    # get start and end date of storm from BEST track
    plot_date_str = storm_best_track.start_date.isoformat().translate({ord(i): None for i in '-:'})
    storm_start_date = storm_best_track.start_date
    storm_end_date = storm_best_track.end_date      
    
    # define date range for plots and stats (+/2 days of a given target date) 
    target_start_date, target_end_date = get_time_period_of_interest(target_date, 2)

    if target_start_date < storm_start_date:
        raise ValueError(f'mid_date should be at least 2 days AFTER {storm_start_date}')
    if target_end_date > storm_end_date:
        raise ValueError(f'mid_date should be at least 2 days BEFORE {storm_end_date}')
        
    # plot isobars
    plot_isobar_domains(storm_best_track, plot_date_str, save_directory, gdf_countries)
    
    # define COOPS domain 
    isotach_limit = 50
    isotach_domain = MultiPolygon(list(polygonize(storm_best_track.wind_swaths(wind_speed=isotach_limit)['BEST'][plot_date_str].exterior)))

    # Retrieve and save CO-OPS tidal predictions
    # _logger.info('Downloading COOPS data within 50kt isotach from searvey ...')
    domain_stations = coops_stations_within_region(region=isotach_domain)
    domain_stations = domain_stations.reset_index()
    
    coops_start_date = format_datetime_for_coops(storm_start_date)
    coops_end_date = format_datetime_for_coops(storm_end_date)
   
    coops_tidal = coops_product_within_region('predictions', 
                                              region=isotach_domain, 
                                              start_date=coops_start_date, 
                                              end_date=coops_end_date)
    # drop NAN variables
    coops_tidal = coops_tidal.drop_vars(['f','s','q'])
    station_ids = coops_tidal['nos_id'].values

    plot_coops_stations(name_of_storm,  year_of_storm, coops_tidal, save_directory, gdf_countries)

    # convert COOPS nc to df and adjust water level
    df_tidal = coops_dataset_to_dataframe(coops_tidal) #, stations=selected_coops_stations)
    df_tidal_adjusted = adjust_coops_water_level(df_tidal)
   
    # Read mesh hgrid and find indices corresponding to COOPS stations
    hgrid_file = Hgrid.open(hgrid_file_path, crs=4326)
    stations_coordinates = get_stations_coordinates(coops_tidal)    
    stations_indices = find_stations_indices(stations_coordinates, hgrid_file) 

    # Read SCHISM nc and convert it to df
    ds_schism = xr.open_dataset(output_file_path)
    df_schism = get_schism_elevation_df(ds_schism, stations_indices, station_ids)
    
    # calculate statistics of obs. and sim. tidal for the target date range
    df_statistics = calc_stats(df_tidal_adjusted.loc[target_start_date:target_end_date], 
                               df_schism.loc[target_start_date:target_end_date])
    
    # plot timeseries, scatter plots with stats, and taylor diagram
    plot_timeseries(df_tidal_adjusted.loc[target_start_date:target_end_date], 
                    df_schism.loc[target_start_date:target_end_date], 
                    domain_stations, name_of_storm, year_of_storm, save_directory)
    
    plot_scatter(df_tidal_adjusted.loc[target_start_date:target_end_date], 
                 df_schism.loc[target_start_date:target_end_date], 
                 df_statistics, domain_stations, name_of_storm,  year_of_storm, save_directory)

    plot_taylor_diagram(df_statistics, name_of_storm,  year_of_storm, save_directory)
    

def entry():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--storm_name", help="name of the storm", type=str)

    parser.add_argument(
        "--storm_year", help="year of the storm", type=int)
    
    parser.add_argument(
        "--mid_date", help="center date for plots and stats:'YYYY-MM-DD'", type=str)
    
    parser.add_argument(
        "--grid_dir", help="path to hgrid.gr3 file", type=str) 
    
    parser.add_argument(
        "--output_dir", help="Path to SCHISM outputs", type=str)

    # optional 
    parser.add_argument(
        "--save_dir", help="directory for saving analysis", default=os.getcwd(), type=str)

    main(parser.parse_args())

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    entry()